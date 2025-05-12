//@ run-pass

#![feature(arbitrary_self_types)]
#![feature(arbitrary_self_types_pointers)]

pub struct A;

impl A {
    pub fn f(self: *const MyNonNull<Self>) -> i32 { 1 }
}

pub struct MyNonNull<T>(T);

impl<T> core::ops::Receiver for MyNonNull<T> {
    type Target = T;
}

impl<T> MyNonNull<T> {
    // Imagine this a NEW method in B<T> shadowing an EXISTING
    // method in A.
    pub fn f(self: *mut Self) -> i32 {
        2
    }
}

fn main() {
    let mut b = MyNonNull(A);
    let b = &mut b;
    let b = b as *mut MyNonNull<A>;
    // We actually allow the shadowing in the case of const vs mut raw
    // pointer receivers.
    assert_eq!(b.f(), 2);
}
