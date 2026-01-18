#![feature(test)]

#[should_panic]
//~^ ERROR `#[should_panic]` should only be applied to functions annotated with `#[test]` or `#[bench]`
pub fn foo() {}


#[test]
#[should_panic]
pub fn bar() {}

#[should_panic]
#[bench]
pub fn bazz() {}

fn main() { }
