use super::{Lrc, Lock};

pub trait Worker: super::Send {
    type Message: super::Send;
    type Result: super::Send;

    fn message(&mut self, msg: Self::Message);
    fn complete(self) -> Self::Result;
}

pub use executor::WorkerExecutor;

#[cfg(parallel_compiler)]
mod executor {
    use super::*;
    use crate::jobserver;
    use parking_lot::Condvar;
    use std::mem;

    struct WorkerQueue<T: Worker> {
        scheduled: bool,
        complete: bool,
        messages: Vec<T::Message>,
        result: Option<T::Result>,
    }

    /// Allows executing a worker on any Rayon thread,
    /// sending it messages and waiting for it to complete its computation.
    pub struct WorkerExecutor<T: Worker> {
        queue: Lock<WorkerQueue<T>>,
        worker: Lock<Option<T>>,
        #[cfg(parallel_compiler)]
        cond_var: Condvar,
    }

    impl<T: Worker> WorkerExecutor<T> {
        pub fn new(worker: T) -> Self {
            WorkerExecutor {
                queue: Lock::new(WorkerQueue {
                    scheduled: false,
                    complete: false,
                    messages: Vec::new(),
                    result: None,
                }),
                worker: Lock::new(Some(worker)),
                #[cfg(parallel_compiler)]
                cond_var: Condvar::new(),
            }
        }

        fn try_run_worker(&self) {
            if let Some(mut worker) = self.worker.try_lock() {
                self.run_worker(&mut *worker);
            }
        }

        fn run_worker(&self, worker: &mut Option<T>) {
            let worker_ref = if let Some(worker_ref) = worker.as_mut() {
                worker_ref
            } else {
                return
            };
            loop {
                let msgs = {
                    let mut queue = self.queue.lock();
                    let msgs = mem::replace(&mut queue.messages, Vec::new());
                    if msgs.is_empty() {
                        queue.scheduled = false;
                        if queue.complete {
                            queue.result = Some(worker.take().unwrap().complete());
                            self.cond_var.notify_all();
                        }
                        break;
                    }
                    msgs
                };
                for msg in msgs {
                    worker_ref.message(msg);
                }
            }
        }

        pub fn complete(&self) -> T::Result {
            let mut queue = self.queue.lock();
            assert!(!queue.complete);
            queue.complete = true;
            if !queue.scheduled {
                // The worker is not scheduled to run, just run it on the current thread.
                queue.scheduled = true;
                mem::drop(queue);
                self.run_worker(&mut *self.worker.lock());
                queue = self.queue.lock();
            } else if let Some(mut worker) = self.worker.try_lock() {
                // Try to run the worker on the current thread.
                // It was scheduled to run, but it may not have started yet.
                // If we are using a single thread, it may never start at all.
                mem::drop(queue);
                self.run_worker(&mut *worker);
                queue = self.queue.lock();
            } else {
                // The worker must be running on some other thread,
                // and will eventually look at the queue again, since queue.scheduled is true.
                // Wait for it.

                #[cfg(parallel_compiler)]
                {
                    // Wait for the result
                    jobserver::release_thread();
                    self.cond_var.wait(&mut queue);
                    jobserver::acquire_thread();
                }
            }
            queue.result.take().unwrap()
        }

        fn queue_message(&self, msg: T::Message) -> bool {
            let mut queue = self.queue.lock();
            queue.messages.push(msg);
            let was_scheduled = queue.scheduled;
            if !was_scheduled {
                queue.scheduled = true;
            }
            was_scheduled
        }

        pub fn message_in_pool(self: &Lrc<Self>, msg: T::Message)
        where
            T: 'static
        {
            if !self.queue_message(msg) {
                let this = self.clone();
                #[cfg(parallel_compiler)]
                rayon::spawn(move || this.try_run_worker());
                #[cfg(not(parallel_compiler))]
                this.try_run_worker();
            }
        }
    }
}

#[cfg(not(parallel_compiler))]
mod executor {
    use super::*;

    pub struct WorkerExecutor<T: Worker> {
        worker: Lock<Option<T>>,
    }

    impl<T: Worker> WorkerExecutor<T> {
        pub fn new(worker: T) -> Self {
            WorkerExecutor {
                worker: Lock::new(Some(worker)),
            }
        }

        #[inline]
        pub fn complete(&self) -> T::Result {
            self.worker.lock().take().unwrap().complete()
        }

        #[inline]
        pub fn message_in_pool(self: &Lrc<Self>, msg: T::Message)
        where
            T: 'static
        {
            self.worker.lock().as_mut().unwrap().message(msg);
        }
    }
}