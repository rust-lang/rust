//@ check-pass

#[should_panic]
//~^ WARN `#[should_panic]` should only be applied to `#[test]` functions
//~| WARN this was previously accepted
pub fn foo() {}


#[test]
#[should_panic]
pub fn bar() {}

fn main() { }
