// edition:2018
// aux-build:trait-import-suggestions.rs
// compile-flags:--extern trait-import-suggestions

mod foo {
    mod foobar {
        pub(crate) trait Foobar {
            fn foobar(&self) { }
        }

        impl Foobar for u32 { }
    }

    pub(crate) trait Bar {
        fn bar(&self) { }
    }

    impl Bar for u32 { }

    fn in_foo() {
        let x: u32 = 22;
        x.foobar(); //~ ERROR no method named `foobar`
    }
}

fn main() {
    let x: u32 = 22;
    x.bar(); //~ ERROR no method named `bar`
    x.baz(); //~ ERROR no method named `baz`
    let y = u32::from_str("33"); //~ ERROR no function or associated item named `from_str`
}
