// Test that parsing can continue after an ill-formed return type

struct A;

impl A {
    fn a() -> return {} //~ ERROR: expected type, found keyword `return`
    fn b(&self) {}
}

struct B;

impl B {
    fn a() -> 1 + 1 { //~ ERROR: expected type, found `1`
        2
    }
    fn b(&self) {}
}

fn main() {
    let a = A;
    a.b();
    let b = A;
    b.b();
}
