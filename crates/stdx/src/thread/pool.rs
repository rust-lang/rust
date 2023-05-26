//! [`Pool`] implements a basic custom thread pool
//! inspired by the [`threadpool` crate](http://docs.rs/threadpool).
//! When you spawn a task you specify a thread intent
//! so the pool can schedule it to run on a thread with that intent.
//! rust-analyzer uses this to prioritize work based on latency requirements.
//!
//! The thread pool is implemented entirely using
//! the threading utilities in [`crate::thread`].

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use crossbeam_channel::{Receiver, Sender};

use super::{Builder, JoinHandle, ThreadIntent};

pub struct Pool {
    // `_handles` is never read: the field is present
    // only for its `Drop` impl.

    // The worker threads exit once the channel closes;
    // make sure to keep `job_sender` above `handles`
    // so that the channel is actually closed
    // before we join the worker threads!
    job_sender: Sender<Job>,
    _handles: Vec<JoinHandle>,
    extant_tasks: Arc<AtomicUsize>,
}

struct Job {
    requested_intent: ThreadIntent,
    f: Box<dyn FnOnce() + Send + 'static>,
}

impl Pool {
    pub fn new(threads: usize) -> Pool {
        const STACK_SIZE: usize = 8 * 1024 * 1024;
        const INITIAL_INTENT: ThreadIntent = ThreadIntent::Worker;

        let (job_sender, job_receiver) = crossbeam_channel::unbounded();
        let extant_tasks = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::with_capacity(threads);
        for _ in 0..threads {
            let handle = Builder::new(INITIAL_INTENT)
                .stack_size(STACK_SIZE)
                .name("Worker".into())
                .spawn({
                    let extant_tasks = Arc::clone(&extant_tasks);
                    let job_receiver: Receiver<Job> = job_receiver.clone();
                    move || {
                        let mut current_intent = INITIAL_INTENT;
                        for job in job_receiver {
                            if job.requested_intent != current_intent {
                                job.requested_intent.apply_to_current_thread();
                                current_intent = job.requested_intent;
                            }
                            extant_tasks.fetch_add(1, Ordering::SeqCst);
                            (job.f)();
                            extant_tasks.fetch_sub(1, Ordering::SeqCst);
                        }
                    }
                })
                .expect("failed to spawn thread");

            handles.push(handle);
        }

        Pool { _handles: handles, extant_tasks, job_sender }
    }

    pub fn spawn<F>(&self, intent: ThreadIntent, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let f = Box::new(move || {
            if cfg!(debug_assertions) {
                intent.assert_is_used_on_current_thread();
            }
            f()
        });

        let job = Job { requested_intent: intent, f };
        self.job_sender.send(job).unwrap();
    }

    pub fn len(&self) -> usize {
        self.extant_tasks.load(Ordering::SeqCst)
    }
}
