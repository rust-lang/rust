// aux-build:noexporttypelib.rs

extern mod noexporttypelib;

fn main() {
    // Here, the type returned by foo() is not exported.
    // This used to cause internal errors when serializing
    // because the def_id associated with the type was
    // not convertible to a path.
  let x: int = noexporttypelib::foo();
    //~^ ERROR expected `int` but found `core::option::Option<int>`
}

