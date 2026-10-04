// https://github.com/rust-lang/rust/issues/16265
#![crate_name="issue_16265_1"]

pub struct Foo;

//@ hasraw issue_16265_1/traits/index.html 'source'
pub mod traits {
    impl PartialEq for super::Foo {
        fn eq(&self, _: &super::Foo) -> bool {
            true
        }
    }
}
