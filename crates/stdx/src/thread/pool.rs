//! [`Pool`] implements a basic custom thread pool
//! inspired by the [`threadpool` crate](http://docs.rs/threadpool).
//! It allows the spawning of tasks under different QoS classes.
//! rust-analyzer uses this to prioritize work based on latency requirements.
//!
//! The thread pool is implemented entirely using
//! the threading utilities in [`crate::thread`].

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use crossbeam_channel::{Receiver, Sender};

use super::{
    get_current_thread_qos_class, set_current_thread_qos_class, Builder, JoinHandle, QoSClass,
    IS_QOS_AVAILABLE,
};

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
    requested_qos_class: QoSClass,
    f: Box<dyn FnOnce() + Send + 'static>,
}

impl Pool {
    pub fn new(threads: usize) -> Pool {
        const STACK_SIZE: usize = 8 * 1024 * 1024;
        const INITIAL_QOS_CLASS: QoSClass = QoSClass::Utility;

        let (job_sender, job_receiver) = crossbeam_channel::unbounded();
        let extant_tasks = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::with_capacity(threads);
        for _ in 0..threads {
            let handle = Builder::new(INITIAL_QOS_CLASS)
                .stack_size(STACK_SIZE)
                .name("Worker".into())
                .spawn({
                    let extant_tasks = Arc::clone(&extant_tasks);
                    let job_receiver: Receiver<Job> = job_receiver.clone();
                    move || {
                        let mut current_qos_class = INITIAL_QOS_CLASS;
                        for job in job_receiver {
                            if job.requested_qos_class != current_qos_class {
                                set_current_thread_qos_class(job.requested_qos_class);
                                current_qos_class = job.requested_qos_class;
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

    pub fn spawn<F>(&self, qos_class: QoSClass, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let f = Box::new(move || {
            if IS_QOS_AVAILABLE {
                debug_assert_eq!(get_current_thread_qos_class(), Some(qos_class));
            }

            f()
        });

        let job = Job { requested_qos_class: qos_class, f };
        self.job_sender.send(job).unwrap();
    }

    pub fn len(&self) -> usize {
        self.extant_tasks.load(Ordering::SeqCst)
    }
}
