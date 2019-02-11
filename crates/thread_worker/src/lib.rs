//! Small utility to correctly spawn crossbeam-channel based worker threads.

use std::thread;

use crossbeam_channel::{bounded, unbounded, Receiver, Sender, RecvError, SendError};
use drop_bomb::DropBomb;

pub struct Worker<I, O> {
    pub inp: Sender<I>,
    pub out: Receiver<O>,
}

pub struct WorkerHandle {
    name: &'static str,
    thread: thread::JoinHandle<()>,
    bomb: DropBomb,
}

pub fn spawn<I, O, F>(name: &'static str, buf: usize, f: F) -> (Worker<I, O>, WorkerHandle)
where
    F: FnOnce(Receiver<I>, Sender<O>) + Send + 'static,
    I: Send + 'static,
    O: Send + 'static,
{
    let (worker, inp_r, out_s) = worker_chan(buf);
    let watcher = WorkerHandle::spawn(name, move || f(inp_r, out_s));
    (worker, watcher)
}

impl<I, O> Worker<I, O> {
    /// Stops the worker. Returns the message receiver to fetch results which
    /// have become ready before the worker is stopped.
    pub fn shutdown(self) -> Receiver<O> {
        self.out
    }

    pub fn send(&self, item: I) -> Result<(), SendError<I>> {
        self.inp.send(item)
    }
    pub fn recv(&self) -> Result<O, RecvError> {
        self.out.recv()
    }
}

impl WorkerHandle {
    fn spawn(name: &'static str, f: impl FnOnce() + Send + 'static) -> WorkerHandle {
        let thread = thread::spawn(f);
        WorkerHandle {
            name,
            thread,
            bomb: DropBomb::new(format!("WorkerHandle {} was not shutdown", name)),
        }
    }

    pub fn shutdown(mut self) -> thread::Result<()> {
        log::info!("waiting for {} to finish ...", self.name);
        let name = self.name;
        self.bomb.defuse();
        let res = self.thread.join();
        match &res {
            Ok(()) => log::info!("... {} terminated with ok", name),
            Err(_) => log::error!("... {} terminated with err", name),
        }
        res
    }
}

/// Sets up worker channels in a deadlock-avoiding way.
/// If one sets both input and output buffers to a fixed size,
/// a worker might get stuck.
fn worker_chan<I, O>(buf: usize) -> (Worker<I, O>, Receiver<I>, Sender<O>) {
    let (input_sender, input_receiver) = bounded::<I>(buf);
    let (output_sender, output_receiver) = unbounded::<O>();
    (Worker { inp: input_sender, out: output_receiver }, input_receiver, output_sender)
}
