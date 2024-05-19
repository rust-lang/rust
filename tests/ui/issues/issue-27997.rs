//@ run-pass
use std::sync::atomic::{Ordering, AtomicUsize};

use std::mem;
struct S<U,V> {
    _u: U,
    size_of_u: usize,
    _v: V,
    size_of_v: usize
}

impl<U, V> S<U, V> {
    fn new(u: U, v: V) -> Self {
        S {
            _u: u,
            size_of_u: mem::size_of::<U>(),
            _v: v,
            size_of_v: mem::size_of::<V>()
        }
    }
}

static COUNT: AtomicUsize = AtomicUsize::new(0);

impl<V, U> Drop for S<U, V> {
    fn drop(&mut self) {
        assert_eq!(mem::size_of::<U>(), self.size_of_u);
        assert_eq!(mem::size_of::<V>(), self.size_of_v);
        COUNT.store(COUNT.load(Ordering::SeqCst)+1, Ordering::SeqCst);
    }
}

fn main() {
    assert_eq!(COUNT.load(Ordering::SeqCst), 0);
    { S::new(0u8, 1u16); }
    assert_eq!(COUNT.load(Ordering::SeqCst), 1);
}
