#![deny(unused)]

struct F; //~ ERROR struct `F` is never constructed
struct B; //~ ERROR struct `B` is never constructed

enum E {
    //~^ ERROR enum `E` is never used
    Foo(F),
    Bar(B),
}

fn main() {}
