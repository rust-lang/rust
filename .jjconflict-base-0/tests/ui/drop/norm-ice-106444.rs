// issue: rust-lang/rust#106444
// ICE failed to normalize
//@ compile-flags: -Zmir-opt-level=3
//@ check-pass

#![crate_type="lib"]

pub trait A {
    type B;
}

pub struct S<T: A>(T::B);

pub fn foo<T: A>(p: *mut S<T>) {
    unsafe { core::ptr::drop_in_place(p) };
}
