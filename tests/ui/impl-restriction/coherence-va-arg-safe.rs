//@ revisions: classic coherence next
//@[classic] compile-flags: -Znext-solver=no
//@[coherence] compile-flags: -Znext-solver=coherence
//@[next] compile-flags: -Znext-solver=globally

#![feature(c_variadic_va_arg_safe)]

use std::ffi::VaArgSafe;

struct Local;

trait LocalTrait {}

impl<T: VaArgSafe> LocalTrait for T {}

impl LocalTrait for Local {}
//~^ ERROR conflicting implementations of trait `LocalTrait` for type `Local`

fn main() {}
