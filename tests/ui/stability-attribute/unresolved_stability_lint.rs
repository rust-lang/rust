#![feature(staged_api)]
#![stable(feature = "uwu", since = "1.0.0")]

#[unstable(feature = "foo", issue = "none")]
impl Foo for () {}
//~^ ERROR cannot find trait `Foo` in this scope

fn main() {}
