//@ check-pass

// Test that we use `sup` not `eq` during method probe, since this has an effect
// on the leak check. This is (conceptually) minimized from a crater run for
// `wrend 0.3.6`.

use std::ops::Deref;

struct A;

impl Deref for A {
    type Target = B<dyn Fn(&())>;

    fn deref(&self) -> &<Self as Deref>::Target { todo!() }
}

struct B<T: ?Sized>(T);
impl<T> B<dyn Fn(T)> {
    fn method(&self) {}
}

fn main() {
    A.method();
}
