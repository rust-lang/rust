//@ compile-flags:--remap-path-prefix={{src-base}}=/does-not-exist

#![doc(html_root_url = "https://example.com/")]

#[macro_export]
macro_rules! make_foo {
    () => {
        pub struct Foo;
        impl Foo {
            pub fn new() -> Foo {
                Foo
            }
        }
    }
}
