#![crate_type = "lib"]

// Test that if any variant is non-unit,
// we need a repr.
enum Enum {
    //~^ ERROR `#[repr(inttype)]` must be specified
    Unit = 1,
    Tuple(),
    Struct {},
}

// Test that if any non-unit variant has an explicit
// discriminant we need a repr.
enum Enum2 {
    //~^ ERROR `#[repr(inttype)]` must be specified
    Tuple() = 2,
}
