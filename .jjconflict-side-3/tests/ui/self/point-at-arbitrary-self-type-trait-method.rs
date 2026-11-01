trait B { fn foo(self: Box<Self>); }
struct A;

impl B for A {
    fn foo(self: Box<Self>) {}
}

fn main() {
    A.foo() //~ ERROR E0599
}
