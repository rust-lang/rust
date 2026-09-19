//@ aux-build:issue-3907.rs

extern crate issue_3907;

type Foo = dyn issue_3907::Foo + 'static; //~ ERROR not dyn compatible [E0038]

struct S {
    name: isize
}

fn bar(_x: Foo) {}
//~^ ERROR not dyn compatible [E0038]
//~| ERROR cannot be known at compilation time [E0277]

fn main() {}
