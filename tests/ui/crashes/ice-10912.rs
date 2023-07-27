#![warn(clippy::unreadable_literal)]
fn f2() -> impl Sized { && 3.14159265358979323846E }
//@no-rustfix
fn main() {}
