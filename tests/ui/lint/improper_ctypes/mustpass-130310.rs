//@ check-pass

//! this test checks that irregular recursive types do not cause stack overflow in ImproperCTypes

use std::marker::PhantomData;

#[repr(C)]
struct A<T> {
    a: *const A<A<T>>,  // without a recursion limit, checking this ends up creating checks for
                        // infinitely deep types the likes of `A<A<A<A<A<A<...>>>>>>`
    p: PhantomData<T>,
}

extern "C" {
    fn f(a: *const A<()>);
}

fn main() {}
