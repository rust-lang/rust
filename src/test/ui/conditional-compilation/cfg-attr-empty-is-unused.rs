// Check that `#[cfg_attr($PREDICATE,)]` triggers the `unused_attribute` lint.

// compile-flags: --cfg TRUE

#![deny(unused)]

#[cfg_attr(FALSE,)] //~ ERROR unused attribute
fn _f() {}

#[cfg_attr(TRUE,)] //~ ERROR unused attribute
fn _g() {}

fn main() {}
