// Using #[macro_export] on a decl macro has no effect and should warn

#![feature(decl_macro)]

#[macro_export] //~ ERROR `#[macro_export]` is not allowed on declarative macro definitions
pub macro foo() {}

// `#[macro_export]` on `macro_rules!` is valid and should not emit unused_attributes
#[macro_export]
macro_rules! bar {
    () => {};
}

fn main() {}
