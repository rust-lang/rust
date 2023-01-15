#![crate_type = "lib"]

pub trait A {
    type B;
}

pub struct S<T: A>(T::B);

pub fn foo<T: A>(p: *mut S<T>) {
    // Verify that we do not ICE when elaborating `Drop(*p)`.
    unsafe { core::ptr::drop_in_place(p) };
}

pub fn bar<U, T: A<B = U>>(p: *mut S<T>) {
    // Verify that we use the correct type for `(*p).0` when elaborating `Drop(*p)`.
    unsafe { core::ptr::drop_in_place(p) };
}

// EMIT_MIR issue_106444.foo.Inline.diff
// EMIT_MIR issue_106444.bar.Inline.diff
