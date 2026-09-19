mod rustfmt {}

#[rustfmt::skip]
//~^ ERROR attribute macro `rustfmt::skip` is ambiguous
//~| ERROR: cannot find `skip` in `rustfmt`
fn main() {}
