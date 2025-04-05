//@ aux-build:noexporttypelib.rs

extern crate noexporttypelib;

fn main() {
    // Here, the type returned by foo() is not exported.
    // This used to cause internal errors when serializing
    // because the def_id associated with the type was
    // not convertible to a path.
  let x: isize = noexporttypelib::foo();
    //~^ ERROR mismatched types
    //~| NOTE expected type `isize`
    //~| NOTE found enum `Option<isize>`
    //~| NOTE expected `isize`, found `Option<isize>`
    //~| NOTE expected due to this
}
