import mne
from mne.time_frequency import tfr_morlet
import numpy as np
from subprocess import PIPE, run
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import wavfile
from scipy.signal import hilbert, chirp, butter, lfilter, freqz
from scipy.io.wavfile import read, write
from scipy import stats
from subprocess import PIPE, run
import pickle
import json
import pandas as pd
import random

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def bin_time(x_idx, sfreq):
    # returns time in seconds
    return float(x_idx / sfreq)

def time_bin(time, sfreq):
    # time must be in seconds
    return int(time*sfreq)

def make_session_template(sub, ses, task):
    sesdir = '../sub-%s/ses-%s/'%(sub,ses)
    stimdir = '../stimuli/all_stimuli/'
    fileid = 'sub-%s_ses-%s_task-%s'%(sub,ses,task)
    
    # load the patient log
    log = pd.read_csv('%s/%s_log.tsv'%(sesdir,fileid) ,sep='\t',header=0)

    # get a list of the times files began playing
    file_start = []
    for index, line in log.iterrows():
        if line[1].startswith('Played catalan'):
            filename = line[1][7:18]
            file_start.append([filename, line[0]])
        elif 'catalan' in line[1]:
            filename = 'pleasePressSpace'
            file_start.append([filename, line[0]])
        elif line[1].startswith('Played excerpts'):
            filename = line[1][-22:-11]
            file_start.append([filename, line[0]])
        elif 'welcome' in line[1]:
            filename = 'welcome'
            file_start.append([filename, line[0]])
        elif 'instructions1' in line[1]:
            filename = 'instructions1'
            file_start.append([filename, line[0]])
        elif 'instructions2' in line[1]:
            filename = 'instructions2'
            file_start.append([filename, line[0]])
        elif 'instructions3' in line[1]:
            filename = 'instructions3'
            file_start.append([filename, line[0]])
        elif 'pleasePressSpace' in line[1]:
            filename = 'pleasePressSpace'
            file_start.append([filename, line[0]])
            
    fs, a = read('%s%s_normed.wav'%(stimdir,file_start[-1][0]))
    lt = file_start[-1][1]
    print(lt, fs)
    seslen = fs*lt + len(a)
    print(seslen)
    template = np.zeros(int(seslen)+1)
    for line in file_start:
        fs, a = read('%s%s_normed.wav'%(stimdir,line[0]))
        #print(line, np.shape(a), fs)
        ssamp = int(fs*line[1])
        #print(ssamp)
        template[ssamp:ssamp+len(a)] = a
        
    return template, fs

def resample(audio_array, audio_rate, new_rate):
    """returns array with audio_rate/new_rate * len(audio_array) samples"""
    cmd = "sox -N -V1 -t f32 -r %s -c 1 - -t f32 -r %s -c 1 -" % (audio_rate, new_rate/2)
    output = run(cmd, shell=True, stdout=PIPE, stderr=PIPE, input=audio_array.tobytes(order="f")).stdout
    audio_data_1 =  np.frombuffer(memoryview(output), dtype=np.float32)
    return np.array([audio_data_1])

def demean_and_norm(audio_array):
    audio_demean = audio_array - (np.mean(audio_array))
    normalization = max(abs(np.amax(audio_demean)),abs(np.amin(audio_demean)))
    audio_normed = (audio_demean / (normalization)).astype(np.float32)
    return audio_array

def load_and_epoch(sub, ses='01', task='PassiveListen', filt=False, bandpower=False, 
                   l_freq=None, h_freq=None, tmin=-.2, tmax=.5, cycle_div = 14.):
    derdir = '/sphere/gentnerlab_NB/acmai/BIDS_PassiveListen/derivatives/preproc-filtering/sub-%s/'%(sub)
    fileid = 'sub-%s_ses-%s_task-%s'%(sub,ses,task)
    fiffile = '%s%s_ica_raw.fif'%(derdir,fileid)
    events_file = '%s%s_epochs.tsv'%(derdir,fileid)
    dict_file = '%s%s_phone_dict.pkl'%(derdir,fileid)
    jsonfile = '/home/acmai/BIDS_PassiveListen/sub-%s/ses-%s/ieeg/%s_ieeg.json'%(sub,ses,fileid)
    annot_file = '%s%s_annotations.csv'%(derdir,fileid)
    
    raw = mne.io.read_raw_fif(fiffile, preload=True)
    neur_rate = raw.info['sfreq']
    info = raw.info

    if filt:
        print("Filtering raw data...")
        raw_filt = mne.filter.filter_data(data=raw.get_data(), sfreq=neur_rate, l_freq=l_freq, h_freq=h_freq)
        raw = mne.io.RawArray(raw_filt, info)
    
    # load annotations and add them to raw
    annots = mne.read_annotations(annot_file)
    raw.set_annotations(annots)
    #print(info['ch_names'][:2])
    #raw.pick_channels(raw.info['ch_names'][:2])
    
    # load phone dictionary & events files
    dtype_dic = {'num': np.int32, 't_start': np.float32, 'seg_prev': str,
            'seg_cur': str, 'phon/wrd/pos/ex/dur': str} 
    event_df = pd.read_csv(events_file, sep='\t', header=0, dtype = dtype_dic)
    event_df['bin'] = (event_df['t_start']*neur_rate).astype(int)
    
    with open(dict_file, 'rb') as dfile:
        dic = pickle.load(dfile)
    
    # map events labels to integers using the phone dictionary
    events = np.array([ [x, 0, dic[y]] for x, y in zip(event_df['bin'], event_df['phon/wrd/pos/ex/dur']) ])
    with info._unlock():
        info['events'] = [{'channels': None, 'list': events}]
    #event_df['events'] = events
    
    dur = tmax - tmin
    e = dur/3.
    # create epochs
    if bandpower:
        print("Creating intial epochs...")
        epochs = mne.Epochs(raw, events, baseline=None, event_id=dic, tmin=tmin-e, tmax=tmax+e, 
                        reject_by_annotation=True, event_repeated='drop', on_missing='warn')
        epochs.drop_bad()
        idxs = epochs.selection.astype(int)
        
        #print(events[idxs][:5])
        if h_freq== 4:
            freqs = np.array([1.5,2.5,3.5])
        else:
            freqs = np.logspace(*np.log10([l_freq,h_freq]), num=8)
        
        n_cycles = freqs / cycle_div  # different number of cycle per frequency
        print("Calculating epoch band power...")
        power = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                return_itc=False, decim=1, n_jobs=1, average=False)
        
        edata = np.mean(power.data, axis=2)
        eb = int(len(epochs.times)/5.)+1
        print("Cropping band power edge artifacts...")
        edata = np.array([x[:,eb:-eb] for x in edata])
        print(edata.shape)
        print("Creating final epochs...")
        epochs = mne.EpochsArray(edata, info=info, events=events[idxs], baseline=None, event_id=dic, 
                                 tmin=tmin, on_missing='warn')
        print(epochs.info['events'])
        
    # create epochs
    else: 
        print("Creating epochs...")
        epochs = mne.Epochs(raw, events, baseline=None, event_id=dic, tmin=tmin, tmax=tmax, 
                        reject_by_annotation=True, event_repeated='drop', on_missing='warn')
        epochs.drop_bad()

    
    print("Done.")
    return raw, neur_rate, epochs


def take_ci(nums, alpha = .95):
    df = np.array(nums).size - 1
    mean = np.mean(nums)
    std = stats.sem(nums)
    # 95% confidence interval
    return stats.t.interval(alpha, df, mean, std)

def epochsCI(epochs, alpha = .95):
    '''
    returns an array of size (n_channels, n_times)
    '''
    epochs_cp = epochs.copy()
    data = epochs_cp.get_data().T
    data = np.transpose(data, (1,0,2))
    #print(data.shape)
    cis = np.array([[take_ci(time, alpha) for time in ch] for ch in data])
    #print(cis.shape)
    return cis

def fill_ci(epochs, ch, ax, alpha = .95):
    '''
    takes an array of size (n_channels, n_times)
    plots a confidence interval for each channel
    '''
    epochs_cp = epochs.copy()
    selection = epochs_cp.load_data().pick_channels(ch).apply_baseline((None,0))
    cis = epochsCI(selection, alpha)
    av = selection.average().data[0]
    for ci in cis:
        #ax.fill_between(selection.times, av + ci, av - ci, alpha=0.3)
        ax.fill_between(selection.times, ci.T[0], ci.T[1], alpha=0.3)
    

def plot_evoked(epochs, ch, ax):
    '''
    ch must be a list
    '''
    epochs_cp = epochs.copy()
    return ax.plot(epochs_cp.times, epochs_cp.load_data().pick_channels(ch).apply_baseline((None,0)).average().data.T)

def plot_zscore_evoked(epochs, ch, ax, tmin, tmax, label=''):
    '''
    ch must be a list (containing one channel)
    shaded SEM
    '''
    epochs_cp = epochs.copy()
    sem = epochs_cp.standard_error()   
    zscored = epochs_cp.apply_baseline((None,0)).get_data() / sem.data
    #print(zscored.shape)

    new = mne.EpochsArray(
            data=zscored, events=epochs_cp.events, 
            event_id=epochs_cp.event_id,info=epochs_cp.info, tmin=tmin,
            baseline=(tmin,0)).crop(tmin,tmax)

    new_line = new.copy().load_data().pick_channels(ch).apply_baseline((None,0)).average().data
    new_sem = new.standard_error().pick_channels(ch).data
    #print(new_line.shape, new_sem.shape)

    sem_lower = new_line - new_sem
    sem_upper = new_line + new_sem

    #print(sem)
    ax.fill_between(new.times, sem_lower[0], sem_upper[0], alpha=0.3)
    ax.plot(new.times, new_line.T, label=label)

def selectRandomSubset(alist, num_to_select):
    '''
    takes a random subset of size num_to_select from 
    a list of objects (alist)
    '''
    try:
        return random.sample(alist,k=num_to_select)
    except:
        if num_to_select > len(alist):
            print("num_to_select > len(alist)")
        else: print("An exception occurred")


def slidingWindowANOVA_reviewers(epochs_list, window = .1, step = 0.05, alpha = 0.05, baseline='condition-specific', bmin=-0.5, bmax=0):
    """
    window and step should be in seconds
    epochs_list is a list of Epochs objects
        Epochs objects should be different conditions from the same patient, session, etc.
        They should have the same channels, same duration, etc.
    baseline can be 'condition-specific' or 'condition-average'
    
    Returns a dictionary of dictionaries, where the top-level dictionary
    is a dictionary of channels, and the lower-level dictionary has
    the following keys:
        'window_size': size of window in seconds
        'times': the start time of each window
        'bf_thresh': alpha bonferroni-corrected for number of windows
        'T': a list containing the T-values for each window
        'p': a list containing the p-values for each window
    """
    ep_tmin = epochs_list[0].times[0]
    dur = epochs_list[0].times[-1] - epochs_list[0].times[0]
    print("duration in seconds: ",dur)
    
    num_wins = int(dur/step)
    threshold_bonferroni = alpha / num_wins
    
    # first make copies of the objects
    epochs_list = [ep.copy() for ep in epochs_list]
    
    # make sure epochs are baselined
    if baseline=='condition-specific':
        epochs_list = specificBaseline(epochs_list, bmin, bmax)
    if baseline=='condition-average':
        epochs_list = averageBaseline(epochs_list, bmin, bmax)
    else: print("ERROR: Invalid baseline parameter.")

    stats_dic = {}
    for ch in epochs_list[0].info['ch_names']:
        print("starting %s...."%(ch))
        eplist_ch = [ep.copy().load_data().pick_channels([ch]) for ep in epochs_list]
        times, Ts, pvals, threshold_bonferronis = [],[],[],[]
        for w in range(num_wins):
            
            #eplist_ch = [ep.load_data().pick_channels([ch]) for ep in epochs_list]
            tmin = w*step+ep_tmin
            tmax = w*step+window+ep_tmin
            eplist_crop = [ep.copy().load_data().crop(tmin=tmin, tmax = tmax).get_data() for ep in eplist_ch]
            
            #average in time
            crop_av = np.array([np.mean(ep, axis=2) for ep in eplist_crop])

            T, pval = stats.f_oneway(*crop_av)
            
            times.append(tmin)
            Ts.append(T)
            pvals.append(pval)
        stats_dic[ch] = {'window_size': window, 'times': times, 'T':Ts, 'p':pvals, 'bf_thresh':threshold_bonferroni}
    
    return stats_dic

def specificBaseline(epochs_list, bmin, bmax):
    #takes a condition-specific baseline for the conditions in epochslist
    baselined_epochs_list = []
    for ep in epochs_list: # for each condition
        means = []
        # for each channel
        for ch in epochs_list[0].info['ch_names']: 
            ch_data = ep.copy().load_data().pick_channels([ch])
            # calculate the mean of the baseline period
            mean_val = np.mean(ch_data.get_data(tmin=bmin, tmax=bmax)) 
            means.append(np.full((1, ep.get_data().shape[2]),mean_val))
        # put all the means in an evoked object
        ep_mean = mne.EvokedArray(np.squeeze(means), ep.copy().info, tmin=bmin) 
        # subtract the mean from the original data
        baselined_epochs_list.append(ep.subtract_evoked(evoked=ep_mean)) 
    return baselined_epochs_list

def averageBaseline(epochs_list, bmin, bmax):
    #takes a condition-average baseline for the conditions in epochslist
    baselined_epochs_list = []
    all_eps = mne.concatenate_epochs(epochs_list) # concatinate all conditions
    means = []
    # for each channel
    for ch in all_eps.info['ch_names']:
        ch_data = all_eps.copy().load_data().pick_channels([ch])
        # calculate the mean of the baseline period
        mean_val = np.mean(ch_data.get_data(tmin=bmin, tmax=bmax))
        means.append(np.full((1, all_eps.get_data().shape[2]),mean_val))
    #put all the means in an evoked object
    ep_mean = mne.EvokedArray(np.squeeze(means), all_eps.copy().info, tmin=bmin)
    #for each condition
    for ep in epochs_list:
        #subtract that mean from the data
        baselined_epochs_list.append(ep.subtract_evoked(evoked=ep_mean))
    return baselined_epochs_list

def invertEventsDic(epochs):
    '''
    epochs is an Epochs object
    returns a dictionary whose keys are event bin numbers
    and whose values are event_labels
    '''
    epochs_cp = epochs.copy()
    return {v: k for k, v in epochs_cp.event_id.items()}

def isWordInitial(event_label, epochs, verbose=False):
    '''
    epochs is an Epochs object
    event_label is a member of epochs.event_id.keys()
    returns boolean
    '''
    epochs_cp = epochs.copy().load_data()
    
    event_num = epochs_cp.event_id[event_label]
    if verbose: print('event_label: ',event_label)
    
    word = event_label.split('/')[1]
    
    #sort events by bin number
    events = epochs_cp.events
    events = events[events[:,0].argsort()]

    event = [(events[x], x) for x in range(len(events)) if events[x,2]==event_num]
    if len(event) > 1:
        print('ERROR: duplicate events')
        return False
    else:
        event, event_index = event[0]
    if verbose: print('event: ', event)
    
    try:
        previous_event = events[event_index-1]
        if verbose: print('previous_event: ', previous_event)
    except Exception as e:
        print(e) # if there is no "previous event"
        return False
    
    rev_dic = invertEventsDic(epochs_cp)
    previous_label = rev_dic[previous_event[2]]
    if verbose: print('previous_label: ',previous_label)
    previous_word = rev_dic[previous_event[2]].split('/')[1]
    if word == previous_word:
        if verbose: print('Not Initial.')
        return False
    else: 
        if verbose: print('INITIAL')
        return True

def isWordFinal(event_label, epochs, verbose=False):
    '''
    epochs is an Epochs object
    event_label is a member of epochs.event_id.keys()
    returns boolean
    '''
    epochs_cp = epochs.copy().load_data()
    
    event_num = epochs_cp.event_id[event_label]
    if verbose: print('event_label: ',event_label)
    
    word = event_label.split('/')[1]
    
    #sort events by bin number
    events = epochs_cp.events
    events = events[events[:,0].argsort()]

    event = [(events[x], x) for x in range(len(events)) if events[x,2]==event_num]
    if len(event) > 1:
        print('ERROR: duplicate events')
        return False
    else:
        event, event_index = event[0]
    if verbose: print('event: ', event)
    
    try:
        next_event = events[event_index+1]
        if verbose: print('next_event: ', next_event)
    except Exception as e:
        print(e) # if there is no "next event"
        return False
    
    rev_dic = invertEventsDic(epochs_cp)
    next_label = rev_dic[next_event[2]]
    if verbose: print('previous_label: ',next_label)
    next_word = rev_dic[next_event[2]].split('/')[1]
    if word == next_word:
        if verbose: print('Not Final.')
        return False
    else: 
        if verbose: print('FINAL')
        return True
    
def bp_from_lfp(lfp, sr, l_freq=70, h_freq=150, cycle_div = 14.):
    '''
    lfp: lfp data as an array (1, n_time_bins)
    sr: lfp sampling rate
    l_freq: low cutoff for bandpass
    h_freq: high cutoff for bandpass
    ----
    returns bandpower data (1, n_time_bins)
    '''
    pl = int(np.array(lfp).shape[1] / 3.) #pad length
    pad = np.zeros((1,pl))
    data = np.hstack([pad, lfp, pad])
    
    info = mne.create_info(ch_names=['broad_lfp'],
                       ch_types=['misc'],
                       sfreq=sr)

    epoch = mne.EpochsArray([data], info)
    
    freqs = np.logspace(*np.log10([l_freq,h_freq]), num=8)
    n_cycles = freqs / cycle_div  # different number of cycle per frequency
    #Calculate band power...
    power = tfr_morlet(epoch, picks=['broad_lfp'], freqs=freqs, n_cycles=n_cycles, use_fft=True,
                            return_itc=False, decim=1, n_jobs=1, average=False)

    edata = np.mean(power.data, axis=2)
    #Crop band power edge artifacts
    edata = np.array([x[:,pl:-pl] for x in edata])
    return edata[0]

def compg(alist, y):
    '''returns boolean for whether list value is greater than y'''
    return [x>y for x in alist]

def compl(alist, y):
    '''returns boolean for whether list value is less than y'''
    return [x<y for x in alist]

def time_idx(dic, time):
    #time should be given in seconds
    #dic is a significance dictionary
    #returns index associated with given time
    ch = list(dic.keys())[0]
    for i,t in enumerate(dic[ch]['times']):
        if t < time: continue
        else: return i
    return i

def confidence_border(matrix, ax, band_dic, band, color, a=0.05):
    prob_dic = {}
    for e in range(matrix.shape[0]):
        prob_dic['etics-%s'%e] = {}
        for o in range (matrix.shape[1]):
            num_both = [[x,y] for [x,y] in band_dic[band]['coord'] if x>=e and y>=o]
            prob_both = len(num_both) / len(band_dic[band]['coord'])
            prob_dic['etics-%s'%e]['ology-%s'%o] = prob_both
            
    pts = []
    for e in range(matrix.shape[0]-1):
        for o in range (matrix.shape[1]-1):
            if prob_dic['etics-%s'%e]['ology-%s'%o] <= a:
                pts.append((e,o))
                break
    for x in range(len(pts)-1):
        if pts[x][1]!=0 or pts[x+1][1]!=0:
            ax.plot([pts[x][1],pts[x+1][1]],[pts[x][0],pts[x+1][0]], c=color, linestyle='--')