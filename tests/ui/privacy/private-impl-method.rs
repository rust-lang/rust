mod a {
    pub struct Foo {
        pub x: isize
    }

    impl Foo {
        fn foo(&self) {}
    }
}

fn f() {
    impl a::Foo {
        fn bar(&self) {} // This should be visible outside `f`
    }
}

fn main() {
    let s = a::Foo { x: 1 };
    s.bar();
    s.foo();    //~ ERROR method `foo` is private
}
