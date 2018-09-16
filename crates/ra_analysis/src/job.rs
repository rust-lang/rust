use crossbeam_channel::{bounded, Receiver, Sender};

pub struct JobHandle {
    job_alive: Receiver<Never>,
    _job_canceled: Sender<Never>,
}

pub struct JobToken {
    _job_alive: Sender<Never>,
    job_canceled: Receiver<Never>,
}

impl JobHandle {
    pub fn new() -> (JobHandle, JobToken) {
        let (sender_alive, receiver_alive) = bounded(0);
        let (sender_canceled, receiver_canceled) = bounded(0);
        let token = JobToken { _job_alive: sender_alive, job_canceled: receiver_canceled };
        let handle = JobHandle { job_alive: receiver_alive, _job_canceled: sender_canceled };
        (handle, token)
    }
    pub fn has_completed(&self) -> bool {
        is_closed(&self.job_alive)
    }
    pub fn cancel(self) {
    }
}

impl JobToken {
    pub fn is_canceled(&self) -> bool {
        is_closed(&self.job_canceled)
    }
}


// We don't actually send messages through the channels,
// and instead just check if the channel is closed,
// so we use uninhabited enum as a message type
enum Never {}

/// Nonblocking
fn is_closed(chan: &Receiver<Never>) -> bool {
    select! {
        recv(chan, msg) => match msg {
            None => true,
            Some(never) => match never {}
        }
        default => false,
    }
}
