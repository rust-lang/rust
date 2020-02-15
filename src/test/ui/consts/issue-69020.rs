// revisions: default noopt opt opt_with_overflow_checks
//[noopt]compile-flags: -C opt-level=0
//[opt]compile-flags: -O
//[opt_with_overflow_checks]compile-flags: -C overflow-checks=on -O

#![crate_type="lib"]

use std::i32;

pub trait Foo {
    const N: i32;
}

impl<T: Foo> Foo for Vec<T> {
    const N: i32 = -i32::MIN + T::N;
    //~^ ERROR arithmetic operation will overflow
}
