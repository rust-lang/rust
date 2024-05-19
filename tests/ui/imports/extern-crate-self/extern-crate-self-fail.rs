extern crate self; //~ ERROR `extern crate self;` requires renaming

#[macro_use] //~ ERROR `#[macro_use]` is not supported on `extern crate self`
extern crate self as foo;

fn main() {}
