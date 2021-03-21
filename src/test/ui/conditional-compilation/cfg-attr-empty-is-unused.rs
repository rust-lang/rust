// Check that `#[cfg_attr($PREDICATE,)]` triggers the `unused_attribute` lint.

// check-pass
// compile-flags: --cfg TRUE

#![deny(unused)]

#[cfg_attr(FALSE,)]
fn _f() {}

#[cfg_attr(TRUE,)]
fn _g() {}

fn main() {}
