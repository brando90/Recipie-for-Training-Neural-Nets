"""
I really liked the trick mentioned on the twitter thread that they train with larger and larger sizes of the data set and increase the model size accordinly too.
Thus, I will create psueod-python code showing the key for this.
The main trick is to change the length of the data set. For my custom data set a had a flag I could change but make sure you do that step right for your data set.
I also added an assert to make sure I never tried doing a data set size that is too large.
"""

def parse_args():
    import argparse

    parser = argparse.ArgumentParser()

    # debugging flags
    parser.add_argument('--train_set_length', type=int, default=236_436)

    opts = parser.parse_args()

    # to have checkpointing work every 6 hours.
    opts.start = time.time()
    opts.next_time_to_ckpt_in_hours = 0.0
    
    return opts
    
def get_dataloader_from_dag_files_syn_sem(opts, rank, world_size, mode):
    """

    Check later
        - if we need pin_memory.
        - why does mnist tutorial have shuffle false?

    :param merge:
    :param opts:
    :param rank:
    :param world_size:
    :return:
    """
    dataloaders = get_dataloaders(opts, mode, rank, world_size, merge_syn_sem)
    return dataloaders

def get_dataloaders(opts, mode, rank, world_size, merge):
    train_dataset = DagDatasetSynSem(opts.path2dataprep, opts.path2hash2idx, 'train', mode)
    assert(opts.train_set_length <= len(train_dataset)) 
    train_dataset.length = opts.train_set_length  ## KEY LINE!!!
    val_dataset = DagDatasetSynSem(opts.path2dataprep, opts.path2hash2idx, 'val', mode)
    test_dataset = DagDatasetSynSem(opts.path2dataprep, opts.path2hash2idx, 'test', mode)
    test_debug_dataset = DagDatasetSynSem(opts.path2dataprep, opts.path2hash2idx, 'test_debug', mode)
    if is_running_serially(rank):
        train_sampler, val_sampler, test_sampler, test_debug_sampler = None, None, None, None
    else:
        assert (opts.batch_size >= world_size)
        # get dist samplers
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
        test_debug_sampler = DistributedSampler(test_debug_dataset, num_replicas=world_size, rank=rank)
    # get dist dataloaders
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=opts.batch_size,
                                  sampler=train_sampler,
                                  collate_fn=merge,
                                  num_workers=0)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=opts.batch_size,
                                sampler=val_sampler,
                                collate_fn=merge,
                                num_workers=0)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=opts.batch_size,
                                 sampler=test_sampler,
                                 collate_fn=merge,
                                 num_workers=0)
    test_debug_dataloader = DataLoader(test_debug_dataset,
                                       batch_size=opts.batch_size,
                                       sampler=test_debug_sampler,
                                       collate_fn=merge,
                                       num_workers=0)
    # return dataloaders
    dataloaders = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader, 'test_debug': test_debug_dataloader}
    return dataloaders

# -- tests

def test_increase_datasize_and_model_size():
    pass

# -- __main__

if __name__ == '__main__':
    test()
