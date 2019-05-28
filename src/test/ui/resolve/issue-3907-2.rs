// aux-build:issue-3907.rs

extern crate issue_3907;

type Foo = dyn issue_3907::Foo + 'static;

struct S {
    name: isize
}

fn bar(_x: Foo) {}
//~^ ERROR E0038

fn main() {}
