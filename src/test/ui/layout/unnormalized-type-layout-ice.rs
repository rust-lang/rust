// Regression test for #85103: do not ICE when failing to normalize
// a type for computing its layout.

#![feature(rustc_attrs)]

trait Foo {
    type Bar;
}

#[rustc_layout(debug)]
type Invalid = <() as Foo>::Bar;
//~^ ERROR layout error: Unknown

fn main() {}
