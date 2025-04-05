use std::any::Any;
use std::mem;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;

use parking_lot::{Condvar, Mutex};

use crate::jobserver;
use crate::sync::{DynSend, FromDyn, IntoDynSyncSend, mode};

enum TaskState<T: DynSend + 'static> {
    Unexecuted(Box<dyn FnOnce() -> T + DynSend>),
    Running,
    Joined,
    Result(Result<T, IntoDynSyncSend<Box<dyn Any + Send + 'static>>>),
}

struct TaskData<T: DynSend + 'static> {
    state: Mutex<TaskState<T>>,
    waiter: Condvar,
}

#[must_use]
pub struct Task<T: DynSend + 'static> {
    data: Arc<TaskData<T>>,
}

/// This attempts to run a closure in a background thread. It returns a `Task` type which
/// you must call `join` on to ensure that the closure runs.
pub fn task<T: DynSend>(f: impl FnOnce() -> T + DynSend + 'static) -> Task<T> {
    let task = Task {
        data: Arc::new(TaskData {
            state: Mutex::new(TaskState::Unexecuted(Box::new(f))),
            waiter: Condvar::new(),
        }),
    };

    if mode::is_dyn_thread_safe() {
        let data = FromDyn::from(Arc::clone(&task.data));

        // Try to execute the task on a separate thread.
        rayon::spawn(move || {
            let data = data.into_inner();
            let mut state = data.state.lock();
            if matches!(*state, TaskState::Unexecuted(..)) {
                if let TaskState::Unexecuted(f) = mem::replace(&mut *state, TaskState::Running) {
                    drop(state);
                    let result = panic::catch_unwind(AssertUnwindSafe(f));

                    let unblock = {
                        let mut state = data.state.lock();
                        let unblock = matches!(*state, TaskState::Joined);
                        *state = TaskState::Result(result.map_err(|e| IntoDynSyncSend(e)));
                        unblock
                    };

                    if unblock {
                        rayon_core::mark_unblocked(&rayon_core::Registry::current());
                    }

                    data.waiter.notify_one();
                }
            }
        });
    }

    task
}

#[inline]
fn unwind<T>(result: Result<T, IntoDynSyncSend<Box<dyn Any + Send + 'static>>>) -> T {
    match result {
        Ok(r) => r,
        Err(err) => panic::resume_unwind(err.0),
    }
}

impl<T: DynSend> Task<T> {
    pub fn join(self) -> T {
        let mut state_guard = self.data.state.lock();

        match mem::replace(&mut *state_guard, TaskState::Joined) {
            TaskState::Unexecuted(f) => f(),
            TaskState::Result(result) => unwind(result),
            TaskState::Running => {
                rayon_core::mark_blocked();
                jobserver::release_thread();

                self.data.waiter.wait(&mut state_guard);

                jobserver::acquire_thread();

                match mem::replace(&mut *state_guard, TaskState::Joined) {
                    TaskState::Result(result) => unwind(result),
                    _ => panic!(),
                }
            }
            TaskState::Joined => panic!(),
        }
    }
}
