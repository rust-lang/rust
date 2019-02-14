//! Small utility to correctly spawn crossbeam-channel based worker threads.

use std::thread;

use crossbeam_channel::{bounded, unbounded, Receiver, Sender};

/// Like `std::thread::JoinHandle<()>`, but joins thread in drop automatically.
pub struct ScopedThread {
    // Option for drop
    inner: Option<thread::JoinHandle<()>>,
}

impl Drop for ScopedThread {
    fn drop(&mut self) {
        let inner = self.inner.take().unwrap();
        let name = inner.thread().name().unwrap().to_string();
        log::info!("waiting for {} to finish...", name);
        let res = inner.join();
        log::info!(".. {} terminated with {}", name, if res.is_ok() { "ok" } else { "err" });

        // escalate panic, but avoid aborting the process
        match res {
            Err(e) => {
                if !thread::panicking() {
                    panic!(e)
                }
            }
            _ => (),
        }
    }
}

impl ScopedThread {
    pub fn spawn(name: &'static str, f: impl FnOnce() + Send + 'static) -> ScopedThread {
        let inner = thread::Builder::new().name(name.into()).spawn(f).unwrap();
        ScopedThread { inner: Some(inner) }
    }
}

/// A wrapper around event-processing thread with automatic shutdown semantics.
pub struct Worker<I, O> {
    // XXX: field order is significant here.
    //
    // In Rust, fields are dropped in the declaration order, and we rely on this
    // here. We must close input first, so that the  `thread` (who holds the
    // opposite side of the channel) noticed shutdown. Then, we must join the
    // thread, but we must keep out alive so that the thread does not panic.
    //
    // Note that a potential problem here is that we might drop some messages
    // from receiver on the floor. This is ok for rust-analyzer: we have only a
    // single client, so, if we are shutting down, nobody is interested in the
    // unfinished work anyway!
    sender: Sender<I>,
    _thread: ScopedThread,
    receiver: Receiver<O>,
}

impl<I, O> Worker<I, O> {
    pub fn spawn<F>(name: &'static str, buf: usize, f: F) -> Worker<I, O>
    where
        F: FnOnce(Receiver<I>, Sender<O>) + Send + 'static,
        I: Send + 'static,
        O: Send + 'static,
    {
        // Set up worker channels in a deadlock-avoiding way. If one sets both input
        // and output buffers to a fixed size, a worker might get stuck.
        let (sender, input_receiver) = bounded::<I>(buf);
        let (output_sender, receiver) = unbounded::<O>();
        let _thread = ScopedThread::spawn(name, move || f(input_receiver, output_sender));
        Worker { sender, _thread, receiver }
    }
}

impl<I, O> Worker<I, O> {
    pub fn sender(&self) -> &Sender<I> {
        &self.sender
    }
    pub fn receiver(&self) -> &Receiver<O> {
        &self.receiver
    }
}
