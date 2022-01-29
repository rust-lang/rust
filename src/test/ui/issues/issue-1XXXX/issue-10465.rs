pub mod a {
    pub trait A {
        fn foo(&self);
    }

}
pub mod b {
    use a::A;

    pub struct B;
    impl A for B { fn foo(&self) {} }

    pub mod c {
        use b::B;

        fn foo(b: &B) {
            b.foo(); //~ ERROR: no method named `foo` found
        }
    }

}

fn main() {}
