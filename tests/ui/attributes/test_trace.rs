//@ compile-flags: --test

fn main() {}

#[test_trace]
//~^ ERROR cannot find attribute `test_trace` in this scope
fn foo() {}


#[test]
#[test_trace]
//~^ ERROR cannot find attribute `test_trace` in this scope
fn bar() {}

#[test]
#[test_trace]
//~^ ERROR cannot find attribute `test_trace` in this scope
#[test]
//~^ WARN duplicated attribute
fn bazz() {}
