//@ only-x86_64
//@ check-pass
use std::sync::atomic::{AtomicUsize, Ordering};

trait Foo {
    fn store(self, ordering: Ordering);
}

impl Foo for AtomicUsize {
    fn store(self, _ordering: Ordering) {
        AtomicUsize::store(&self, 4, Ordering::SeqCst);
    }
}

fn main() {
    let x = AtomicUsize::new(3);
    x.store(Ordering::Acquire);
}
