//! This test checks the one code path that does not go through
//! the regular CTFE memory access (as an optimization). We forgot
//! to duplicate the static item self-initialization check, allowing
//! reading from the uninitialized static memory before it was
//! initialized at the end of the static initializer.
//!
//! https://github.com/rust-lang/rust/issues/142532

use std::mem::MaybeUninit;

pub static X: (i32, MaybeUninit<i32>) = (1, foo(&X.0));
//~^ ERROR: encountered static that tried to initialize itself with itself

const fn foo(x: &i32) -> MaybeUninit<i32> {
    let mut temp = MaybeUninit::<i32>::uninit();
    unsafe {
        std::ptr::copy(x, temp.as_mut_ptr(), 1);
    }
    temp
}

fn main() {}
