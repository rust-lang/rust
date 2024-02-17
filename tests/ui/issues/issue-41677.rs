//@ run-pass
// Regression test for #41677. The local variable was winding up with
// a type `Receiver<?T, H>` where `?T` was unconstrained, because we
// failed to enforce the WF obligations and `?T` is a bivariant type
// parameter position.

#![allow(unused_variables, dead_code)]

use std::marker::PhantomData;

trait Handle {
    type Inner;
}

struct ResizingHandle<H>(PhantomData<H>);
impl<H> Handle for ResizingHandle<H> {
    type Inner = H;
}

struct Receiver<T, H: Handle<Inner=T>>(PhantomData<H>);

fn channel<T>(size: usize) -> Receiver<T, ResizingHandle<T>> {
    let rx = Receiver(PhantomData);
    rx
}

fn main() {
}
