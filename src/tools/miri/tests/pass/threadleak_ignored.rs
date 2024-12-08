//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
//@compile-flags: -Zmiri-ignore-leaks

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

    // Set up a channel so that we can learn when the other thread initialized `X`
    // (so that we are sure there is something to drop).
    let (send, recv) = std::sync::mpsc::channel::<()>();

    let _detached = std::thread::spawn(move || {
        X.with(|x| *x.borrow_mut() = Some(LoudDrop(1)));
        send.send(()).unwrap();
        std::thread::yield_now();
        loop {}
    });

    std::thread::yield_now();

    // Wait until child thread has initialized its `X`.
    let () = recv.recv().unwrap();
}
