// pretty-expanded FIXME #23616

pub mod Bar {
    pub struct Foo {
        v: isize,
    }

    extern {
        pub fn foo(v: *const Foo) -> Foo;
    }
}

pub fn main() { }
