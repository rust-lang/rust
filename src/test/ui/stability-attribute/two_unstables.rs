//check-pass

#![unstable(feature = "module", issue = "none")]

#![feature(staged_api)]

#[unstable(feature = "a", issue = "none")]
#[unstable(feature = "b", issue = "none")]
pub struct Foo;

fn main() {}
