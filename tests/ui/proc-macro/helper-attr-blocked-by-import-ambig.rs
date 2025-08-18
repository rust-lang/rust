//@ proc-macro: test-macros.rs

#[macro_use(Empty)]
extern crate test_macros;
use test_macros::empty_attr as empty_helper;

#[empty_helper] //~ ERROR `empty_helper` is ambiguous
                //~| ERROR derive helper attribute is used before it is introduced
                //~| WARN this was previously accepted
#[derive(Empty)]
struct S;

fn main() {}
