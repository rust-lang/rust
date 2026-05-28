#![feature(unboxed_closures)]

//@ check-pass

// Regression test for #131758. We only know the type of `x` after closure upvar
// inference is done, even if we don't need to structurally resolve the type of `x`.

trait Foo {}

impl<T: Fn<(i32,)>> Foo for T {}

fn baz<T: Foo>(_: T) {}

fn main() {
    baz(|x| ());
}
