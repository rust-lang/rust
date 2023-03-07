// Check that `#[cfg_attr($PREDICATE,)]` triggers the `unused_attribute` lint.

// compile-flags: --cfg TRUE

#![deny(unused)]

#[cfg_attr(FALSE,)] //~ ERROR `#[cfg_attr]` does not expand to any attributes
fn _f() {}

#[cfg_attr(TRUE,)] //~ ERROR `#[cfg_attr]` does not expand to any attributes
fn _g() {}

fn main() {}
