// run-pass
use std::sync::atomic::{AtomicUsize, Ordering};

static DROP_COUNTER: AtomicUsize = AtomicUsize::new(0);

struct A(i32);

impl Drop for A {
    fn drop(&mut self) {
        // update global drop count
        DROP_COUNTER.fetch_add(1, Ordering::SeqCst);
    }
}

static FOO: A = A(123);
const BAR: A = A(456);

impl A {
    const BAZ: A = A(789);
}

fn main() {
    assert_eq!(DROP_COUNTER.load(Ordering::SeqCst), 0);
    assert_eq!(&FOO.0, &123);
    assert_eq!(DROP_COUNTER.load(Ordering::SeqCst), 0);
    assert_eq!(BAR.0, 456);
    assert_eq!(DROP_COUNTER.load(Ordering::SeqCst), 1);
    assert_eq!(A::BAZ.0, 789);
    assert_eq!(DROP_COUNTER.load(Ordering::SeqCst), 2);
}
