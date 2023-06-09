struct A;

impl A {
    fn foo(self: Box<Self>) {}
}

fn main() {
    A.foo(); //~ ERROR E0599
}
