// run-pass
// aux-build:issue-19293.rs
// pretty-expanded FIXME #23616

extern crate issue_19293;
use issue_19293::{Foo, MyEnum};

fn main() {
    MyEnum::Foo(Foo(5));
}
