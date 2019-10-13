#![deny(unused)]

struct F; //~ ERROR struct is never constructed
struct B; //~ ERROR struct is never constructed

enum E { //~ ERROR enum is never used
    Foo(F),
    Bar(B),
}

fn main() {}
