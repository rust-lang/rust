// Using #[macro_export] on a decl macro has no effect and should warn

#![feature(decl_macro)]
#![deny(unused)]

#[macro_export] //~ ERROR `#[macro_export]` has no effect on declarative macro definitions
pub macro foo() {}

fn main() {}
