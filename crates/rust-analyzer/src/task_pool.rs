//! A thin wrapper around `ThreadPool` to make sure that we join all things
//! properly.
use std::sync::{Arc, Barrier};

use crossbeam_channel::Sender;

pub(crate) struct TaskPool<T> {
    sender: Sender<T>,
    inner: threadpool::ThreadPool,
}

impl<T> TaskPool<T> {
    pub(crate) fn new_with_threads(sender: Sender<T>, threads: usize) -> TaskPool<T> {
        const STACK_SIZE: usize = 8 * 1024 * 1024;

        let inner = threadpool::Builder::new()
            .thread_name("Worker".into())
            .thread_stack_size(STACK_SIZE)
            .num_threads(threads)
            .build();

        // Set QoS of all threads in threadpool.
        let barrier = Arc::new(Barrier::new(threads + 1));
        for _ in 0..threads {
            let barrier = barrier.clone();
            inner.execute(move || {
                stdx::thread::set_current_thread_qos_class(stdx::thread::QoSClass::Utility);
                barrier.wait();
            });
        }
        barrier.wait();

        TaskPool { sender, inner }
    }

    pub(crate) fn spawn<F>(&mut self, task: F)
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        self.inner.execute({
            let sender = self.sender.clone();
            move || {
                if stdx::thread::IS_QOS_AVAILABLE {
                    debug_assert_eq!(
                        stdx::thread::get_current_thread_qos_class(),
                        Some(stdx::thread::QoSClass::Utility)
                    );
                }

                sender.send(task()).unwrap()
            }
        })
    }

    pub(crate) fn spawn_with_sender<F>(&mut self, task: F)
    where
        F: FnOnce(Sender<T>) + Send + 'static,
        T: Send + 'static,
    {
        self.inner.execute({
            let sender = self.sender.clone();
            move || task(sender)
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.inner.queued_count()
    }
}

impl<T> Drop for TaskPool<T> {
    fn drop(&mut self) {
        self.inner.join()
    }
}
