// ignore-windows: Concurrency on Windows is not supported yet.
// compile-flags: -Zmiri-ignore-leaks

//! Test that leaking threads works, and that their destructors are not executed.

use std::cell::RefCell;

struct LoudDrop(i32);
impl Drop for LoudDrop {
    fn drop(&mut self) {
        eprintln!("Dropping {}", self.0);
    }
}

thread_local! {
    static X: RefCell<Option<LoudDrop>> = RefCell::new(None);
}

fn main() {
    X.with(|x| *x.borrow_mut() = Some(LoudDrop(0)));
    
    let _detached = std::thread::spawn(|| {
        X.with(|x| *x.borrow_mut() = Some(LoudDrop(1)));
    });
}
