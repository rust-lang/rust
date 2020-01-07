use crate::sync::{self, FlexScope};
use std::mem::ManuallyDrop;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;
use std::sync::Condvar;
use std::sync::Mutex;
use std::thread;

struct FutureData<'a, R> {
    // ManuallyDrop is needed here to ensure the destructor of FutureData cannot refer to 'a
    func: Mutex<ManuallyDrop<Option<Box<dyn FnOnce() -> R + sync::Send + 'a>>>>,
    result: Mutex<Option<thread::Result<R>>>,
    waiter: Condvar,
}
pub struct Future<'a, R> {
    data: Arc<FutureData<'a, R>>,
}

impl<'a, R: sync::Send + 'a> Future<'a, R> {
    pub fn spawn_in_scope(scope: &FlexScope<'a>, f: impl FnOnce() -> R + sync::Send + 'a) -> Self {
        let data = Arc::new(FutureData {
            func: Mutex::new(ManuallyDrop::new(Some(Box::new(f)))),
            result: Mutex::new(None),
            waiter: Condvar::new(),
        });
        let result = Self { data: data.clone() };
        scope.spawn(move || {
            if let Some(func) = data.func.lock().unwrap().take() {
                // Execute the function if it has not yet been joined
                let r = panic::catch_unwind(AssertUnwindSafe(func));
                *data.result.lock().unwrap() = Some(r);
                data.waiter.notify_all();
            }
        });
        result
    }

    pub fn join(self) -> R {
        if let Some(func) = self.data.func.lock().unwrap().take() {
            // The function was not executed yet by Rayon, just run it
            func()
        } else {
            // The function has started executing, wait for it to complete
            let mut result = self
                .data
                .waiter
                .wait_until(self.data.result.lock().unwrap(), |result| result.is_some())
                .unwrap();
            match result.take().unwrap() {
                Ok(r) => {
                    return r;
                }
                Err(err) => panic::resume_unwind(err),
            }
        }
    }
}
