//@ build-pass
//@ compile-flags: --emit=obj
#![feature(const_c_variadic)]
#![feature(const_destruct)]
#![crate_type = "lib"]

// Regression test for when a c-variadic argument is `PassMode::Ignore`. The caller won't pass the
// argument, but the callee ABI does have the argument. Ensure that const-eval is able to handle
// this case without tripping any asserts.

const unsafe extern "C" fn read_n<const N: usize>(_: ...) {}

unsafe fn read_too_many() {
    const { read_n::<0>((), 1i32) }
}

fn read_as<T>() -> () {}
