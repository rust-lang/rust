use crate::jobserver;
use crate::sync;
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

fn create<'a, R>(f: impl FnOnce() -> R + sync::Send + 'a) -> Future<'a, R> {
    let data = Arc::new(FutureData {
        func: Mutex::new(ManuallyDrop::new(Some(Box::new(f)))),
        result: Mutex::new(None),
        waiter: Condvar::new(),
    });
    Future { data: data.clone() }
}

fn run<R>(data: &FutureData<'_, R>) {
    if let Some(func) = data.func.lock().unwrap().take() {
        // Execute the function if it has not yet been joined
        let r = panic::catch_unwind(AssertUnwindSafe(func));
        *data.result.lock().unwrap() = Some(r);
        data.waiter.notify_all();
    }
}

impl<R: sync::Send + 'static> Future<'static, R> {
    pub fn spawn(f: impl FnOnce() -> R + sync::Send + 'static) -> Self {
        let result = create(f);
        let data = result.data.clone();
        sync::spawn(move || run(&data));
        result
    }
}

impl<'a, R: sync::Send + 'a> Future<'a, R> {
    pub fn join(self) -> R {
        if let Some(func) = self.data.func.lock().unwrap().take() {
            // The function was not executed yet by Rayon, just run it
            func()
        } else {
            // The function has started executing, wait for it to complete
            jobserver::release_thread();
            let mut result = self
                .data
                .waiter
                .wait_while(self.data.result.lock().unwrap(), |result| result.is_none())
                .unwrap();
            jobserver::acquire_thread();
            match result.take().unwrap() {
                Ok(r) => {
                    return r;
                }
                Err(err) => panic::resume_unwind(err),
            }
        }
    }
}
