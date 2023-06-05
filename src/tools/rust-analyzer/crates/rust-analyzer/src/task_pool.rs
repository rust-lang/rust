//! A thin wrapper around [`stdx::thread::Pool`] which threads a sender through spawned jobs.
//! It is used in [`crate::global_state::GlobalState`] throughout the main loop.

use crossbeam_channel::Sender;
use stdx::thread::{Pool, ThreadIntent};

pub(crate) struct TaskPool<T> {
    sender: Sender<T>,
    pool: Pool,
}

impl<T> TaskPool<T> {
    pub(crate) fn new_with_threads(sender: Sender<T>, threads: usize) -> TaskPool<T> {
        TaskPool { sender, pool: Pool::new(threads) }
    }

    pub(crate) fn spawn<F>(&mut self, intent: ThreadIntent, task: F)
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        self.pool.spawn(intent, {
            let sender = self.sender.clone();
            move || sender.send(task()).unwrap()
        })
    }

    pub(crate) fn spawn_with_sender<F>(&mut self, intent: ThreadIntent, task: F)
    where
        F: FnOnce(Sender<T>) + Send + 'static,
        T: Send + 'static,
    {
        self.pool.spawn(intent, {
            let sender = self.sender.clone();
            move || task(sender)
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.pool.len()
    }
}
