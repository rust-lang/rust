//@ aux-build:issue-21202.rs

extern crate issue_21202 as crate1;

use crate1::A;

mod B {
    use crate1::A::Foo;
    fn bar(f: Foo) {
        Foo::foo(&f);
        //~^ ERROR: method `foo` is private
    }
}

fn main() { }
