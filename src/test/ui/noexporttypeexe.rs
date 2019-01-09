// aux-build:noexporttypelib.rs

extern crate noexporttypelib;

fn main() {
    // Here, the type returned by foo() is not exported.
    // This used to cause internal errors when serializing
    // because the def_id associated with the type was
    // not convertible to a path.
  let x: isize = noexporttypelib::foo();
    //~^ ERROR mismatched types
    //~| expected type `isize`
    //~| found type `std::option::Option<isize>`
    //~| expected isize, found enum `std::option::Option`
}
