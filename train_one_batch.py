
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        """
        Note: usually n=batch_size so that we can keep track of the total sum.
        If you don't log the batch size the quantity your tracking is the average of the sample means
        which has the same expectation but because your tracking emperical estimates you will have a
        different variance. Thus, it's recommended to have n=batch_size

        :param val:
        :param n: usually the batch size
        :return:
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def item(self):
        return self.avg

    def __str__(self):
        fmtstr = '{name} val:{val} avg:{avg}'
        return fmtstr.format(**self.__dict__)
        
def train_one_batch(opts, model, train_batch, val_batch, optimizer, tolerance=0.01):
    """
    Code for training a generic pytorch model with one abtch.
    The idea is that the user uses (their perhaps custom) data loader to sample a data batch once
    and then pass them to this code to train until the batch has been overfitted.

    Note: If you are doing regression you will have to adapt this code - however, I do recommend that
    you track some sort of accuracy for your regression task. For example, track R2 (or some squeezed
    version of it). This is really useful because your loss has an arbitrary uninterpretable scale
    while R2 always has a nice interpretation (how far you are from just predicting the mean target y
    of your data without using any features x). Having this sort of interpretable measure can save you
    a lot of time - especially when the loss seems to be meaninfuless.
    For that replace accuracy for your favorite interpretable "acuraccy" function.

    :param opts:
    :param model:
    :param train_batch:
    :param val_batch:
    :param optimizer:
    :param tolerance:
    :return:
    """
    avg_loss = AverageMeter('train loss')
    avg_acc = AverageMeter('train accuracy')

    it = 0
    train_loss = float('inf')
    x_train_batch, y_train_batch = train_batch
    x_val_batch, y_val_batch = val_batch
    while train_loss > tolerance:
        model.train()
        train_loss, logits = model(x_train_batch)
        avg_loss.update(train_loss.item(), opts.batch_size)
        train_acc = accuracy(output=logits, target=y_train_batch)
        avg_acc.update(train_acc.item(), opts.batch_size)

        optimizer.zero_grad()
        train_loss.backward()  # each process synchronizes it's gradients in the backward pass
        optimizer.step()  # the right update is done since all procs have the right synced grads

        model.eval()
        val_loss, logits = model(x_train_batch)
        avg_loss.update(train_loss.item(), opts.batch_size)
        val_acc = accuracy(output=logits, target=y_val_batch)
        avg_acc.update(val_acc.item(), opts.batch_size)

        log_2_tb(it=it, tag1='train loss', loss=train_loss.item(), tag2='train acc', acc=train_acc.item())
        log_2_tb(it=it, tag1='val loss', loss=val_loss.item(), tag2='val acc', acc=val_acc.item())
        print(f"\n{it=}: {train_loss=} {train_acc=}")
        print(f"{it=}: {val_loss=} {val_acc=}")

        it += 1
        gc.collect()

    return avg_loss.item(), avg_acc.item()

# -- tests

def test():
    pass()

# -- __main__

if __name__ == '__main__':
    # test()
    print('Done\a')
