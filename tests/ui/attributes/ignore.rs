#![feature(test)]

#[ignore]
//~^ ERROR `#[ignore]` should only be applied to functions annotated with `#[test]` or `#[bench]`
pub fn foo() {}


#[test]
#[ignore]
pub fn bar() {}

#[ignore]
#[bench]
pub fn bazz() {}

fn main() { }
