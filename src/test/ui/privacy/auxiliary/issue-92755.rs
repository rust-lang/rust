mod machine {
    pub struct A {
        pub b: B,
    }
    pub struct B {}
    impl B {
        pub fn f(&self) {}
    }
}

pub struct Context {
    pub a: machine::A,
}

pub fn ctx() -> Context {
    todo!();
}
