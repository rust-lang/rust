mod m2 {
    pub enum Foo {
        A,
        B(isize),
        C { a: isize },
    }

    impl Foo {
        pub fn foo() {}
        pub fn bar(&self) {}
    }
}

mod m {
    pub use m2::Foo::*;
}

pub fn main() {
    use m2::Foo::*;

    foo(); //~ ERROR cannot find function `foo`
    m::foo(); //~ ERROR cannot find function `foo`
    bar(); //~ ERROR cannot find function `bar`
    m::bar(); //~ ERROR cannot find function `bar`
}
