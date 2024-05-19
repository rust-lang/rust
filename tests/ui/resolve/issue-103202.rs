struct S {}

impl S {
    fn f(self: &S::x) {} //~ ERROR ambiguous associated type
}

fn main() {}
