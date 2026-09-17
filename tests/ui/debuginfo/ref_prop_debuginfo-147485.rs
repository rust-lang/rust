//@ build-pass
//@ compile-flags: -g -O

// Regression test for #147485.

#![crate_type = "lib"]

#[allow(safe_fn_direct_use_of_unsafe_op_on_args)]
pub fn f(x: *const usize) -> &'static usize {
    let mut a = unsafe { &*x };
    a = unsafe { &*x };
    a
}

pub fn g() {
    f(&0);
}
