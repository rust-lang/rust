// check-pass
// FIXME(#54943) This test targets the scenario where proving the WF requirements requires
// knowing the value of the `_` type present in the user type annotation - unfortunately, figuring
// out the value of that `_` requires type-checking the surrounding code, but that code is dead,
// so our NLL region checker doesn't have access to it. This test should actually fail to compile.

#![allow(warnings)]

use std::fmt::Debug;

fn foo<T: 'static + Debug>(_: T) { }

fn bar<'a>() {
    return;

    foo::<Vec<_>>(Vec::<&'a u32>::new());
}

fn main() {}
