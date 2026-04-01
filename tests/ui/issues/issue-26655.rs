//@ run-pass
//@ needs-unwind
//@ needs-threads
//@ ignore-backends: gcc

// Check that the destructors of simple enums are run on unwinding

use std::sync::atomic::{Ordering, AtomicUsize};
use std::thread;

static LOG: AtomicUsize = AtomicUsize::new(0);

enum WithDtor { Val }
impl Drop for WithDtor {
    fn drop(&mut self) {
        LOG.store(LOG.load(Ordering::SeqCst)+1,Ordering::SeqCst);
    }
}

pub fn main() {
    thread::spawn(move|| {
        let _e: WithDtor = WithDtor::Val;
        panic!("fail");
    }).join().unwrap_err();

    assert_eq!(LOG.load(Ordering::SeqCst), 1);
}
