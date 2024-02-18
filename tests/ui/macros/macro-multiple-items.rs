//@ run-pass
macro_rules! make_foo {
    () => (
        struct Foo;

        impl Foo {
            fn bar(&self) {}
        }
    )
}

make_foo!();

pub fn main() {
    Foo.bar()
}
