#![deny(unused)]

struct F;
struct B;

enum E {
    Foo(F),
    Bar(B), //~ ERROR variant `Bar` is never constructed
}

fn main() {
    let _ = E::Foo(F);
}
