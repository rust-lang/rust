// While somewhat nonsensical, this is a cast from a wide pointer to a thin pointer.
// Thus, we don't need to check an unsize goal here; there isn't any vtable casting
// happening at all.

// Regression test for <https://github.com/rust-lang/rust/issues/137579>.

//@ check-pass

#![allow(incomplete_features)]
#![feature(dyn_star)]

trait Foo {}
trait Bar {}

fn cast(x: *const dyn Foo) {
    x as *const dyn* Bar;
}

fn main() {}
