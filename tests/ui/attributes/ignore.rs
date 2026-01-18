//@ check-pass
#![feature(test)]

#[ignore]
//~^ WARN `#[ignore]` should only be applied to functions annotated with `#[test]` or `#[bench]`
//~| WARN this was previously accepted
pub fn foo() {}


#[test]
#[ignore]
pub fn bar() {}

#[ignore]
#[bench]
pub fn bazz() {}

fn main() { }
