#![feature(decl_macro)]
#![feature(pub_macro_rules)]

#[macro_export]
macro m1() {} //~ ERROR `#[macro_export]` cannot be used on `macro` items

#[macro_export]
pub macro_rules! m2 { () => {} }
//~^ ERROR `#[macro_export]` cannot be used on `macro_rules` with `pub`

fn main() {}
