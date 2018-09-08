use std::thread;
use crossbeam_channel::{bounded, unbounded, Sender, Receiver};
use drop_bomb::DropBomb;
use Result;

pub struct ThreadWatcher {
    name: &'static str,
    thread: thread::JoinHandle<()>,
    bomb: DropBomb,
}

impl ThreadWatcher {
    pub fn spawn(name: &'static str, f: impl FnOnce() + Send + 'static) -> ThreadWatcher {
        let thread = thread::spawn(f);
        ThreadWatcher {
            name,
            thread,
            bomb: DropBomb::new(format!("ThreadWatcher {} was not stopped", name)),
        }
    }

    pub fn stop(mut self) -> Result<()> {
        info!("waiting for {} to finish ...", self.name);
        let name = self.name;
        self.bomb.defuse();
        let res = self.thread.join()
            .map_err(|_| format_err!("ThreadWatcher {} died", name));
        match &res {
            Ok(()) => info!("... {} terminated with ok", name),
            Err(_) => error!("... {} terminated with err", name)
        }
        res
    }
}

/// Sets up worker channels in a deadlock-avoind way.
/// If one sets both input and output buffers to a fixed size,
/// a worker might get stuck.
pub fn worker_chan<I, O>(buf: usize) -> ((Sender<I>, Receiver<O>), Receiver<I>, Sender<O>) {
    let (input_sender, input_receiver) = bounded::<I>(buf);
    let (output_sender, output_receiver) = unbounded::<O>();
    ((input_sender, output_receiver), input_receiver, output_sender)
}
