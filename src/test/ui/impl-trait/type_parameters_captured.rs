// ignore-compare-mode-nll
// revisions: base nll
// [nll]compile-flags: -Zborrowck=mir

use std::fmt::Debug;

trait Any {}
impl<T> Any for T {}

// Check that type parameters are captured and not considered 'static
fn foo<T>(x: T) -> impl Any + 'static {
    //[base]~^ ERROR the parameter type `T` may not live long enough
    x
    //[nll]~^ ERROR the parameter type `T` may not live long enough
}

fn main() {}
