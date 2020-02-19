// revisions: noopt opt opt_with_overflow_checks
//[noopt]compile-flags: -C opt-level=0
//[opt]compile-flags: -O
//[opt_with_overflow_checks]compile-flags: -C overflow-checks=on -O

#![crate_type="lib"]

use std::i32;

pub trait Foo {
    const NEG: i32;
    const ADD: i32;
    const DIV: i32;
    const OOB: i32;
}

// These constants cannot be evaluated already (they depend on `T::N`), so
// they can just be linted like normal run-time code.  But codegen works
// a bit different in const context, so this test makes sure that we still catch overflow.
impl<T: Foo> Foo for Vec<T> {
    const NEG: i32 = -i32::MIN + T::NEG;
    //~^ ERROR arithmetic operation will overflow
    const ADD: i32 = (i32::MAX+1) + T::ADD;
    //~^ ERROR arithmetic operation will overflow
    const DIV: i32 = (1/0) + T::DIV;
    //~^ ERROR operation will panic
    const OOB: i32 = [1][1] + T::OOB;
    //~^ ERROR operation will panic
}
