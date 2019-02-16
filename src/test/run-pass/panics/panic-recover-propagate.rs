// run-pass
// ignore-emscripten no threads support

use std::sync::atomic::{AtomicUsize, Ordering};
use std::panic;
use std::thread;

static A: AtomicUsize = AtomicUsize::new(0);

fn main() {
    panic::set_hook(Box::new(|_| {
        A.fetch_add(1, Ordering::SeqCst);
    }));

    let result = thread::spawn(|| {
        let result = panic::catch_unwind(|| {
            panic!("hi there");
        });

        panic::resume_unwind(result.unwrap_err());
    }).join();

    let msg = *result.unwrap_err().downcast::<&'static str>().unwrap();
    assert_eq!("hi there", msg);
    assert_eq!(1, A.load(Ordering::SeqCst));
}
