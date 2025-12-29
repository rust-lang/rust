//@ check-pass
#![feature(test)]

#[should_panic]
//~^ WARN `#[should_panic]` should only be applied to functions annotated with `#[test]` or `#[bench]`
//~| WARN this was previously accepted
pub fn foo() {}


#[test]
#[should_panic]
pub fn bar() {}

#[should_panic]
#[bench]
pub fn bazz() {}

fn main() { }
