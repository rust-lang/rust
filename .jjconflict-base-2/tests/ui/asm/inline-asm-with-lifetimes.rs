//@ revisions: good bad
//@[good] build-pass
//@ needs-asm-support

use std::arch::asm;

// lifetime requirement, we should check it!!
#[cfg(bad)]
fn dep<'a, T: 'a>() {}

// no lifetime requirement
#[cfg(good)]
fn dep<'a: 'a, T>() {}

fn test<'a: 'a, T>() {
    unsafe {
        asm!("/* {} */", sym dep::<'a, T> );
        //[bad]~^ ERROR the parameter type `T` may not live long enough
    }
}

fn main() {}
