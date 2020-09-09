struct S;

impl S {
    fn f() {}
    fn g() {
        use Self::f; //~ ERROR unresolved import
        pub(in Self::f) struct Z; //~ ERROR use of undeclared type `Self`
    }
}

fn main() {}
