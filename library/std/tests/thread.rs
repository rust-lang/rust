use std::cell::{Cell, RefCell};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

#[test]
#[cfg_attr(target_os = "emscripten", ignore)]
fn sleep() {
    let finished = Arc::new(Mutex::new(false));
    let t_finished = finished.clone();
    thread::spawn(move || {
        thread::sleep(Duration::new(u64::MAX, 0));
        *t_finished.lock().unwrap() = true;
    });
    thread::sleep(Duration::from_millis(100));
    assert_eq!(*finished.lock().unwrap(), false);
}

#[test]
fn thread_local_containing_const_statements() {
    // This exercises the `const $init:block` cases of the thread_local macro.
    // Despite overlapping with expression syntax, the `const { ... }` is not
    // parsed as `$init:expr`.
    thread_local! {
        static CELL: Cell<u32> = const {
            let value = 1;
            Cell::new(value)
        };

        static REFCELL: RefCell<u32> = const {
            let value = 1;
            RefCell::new(value)
        };
    }

    assert_eq!(CELL.get(), 1);
    assert_eq!(REFCELL.take(), 1);
}
