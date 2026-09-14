#![feature(rustc_attrs)]

#[inline] //~ NOTE: the inline attribute is specified here
#[rustc_force_inline] //~ ERROR: cannot be used together
fn foo() {}

#[rustc_force_inline] //~ ERROR: cannot be used together
#[inline] //~ NOTE: the inline attribute is specified here
fn bar() {}

fn main() {}
