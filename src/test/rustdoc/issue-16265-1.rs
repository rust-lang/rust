pub struct Foo;

// @has issue_16265_1/traits/index.html 'source'
pub mod traits {
    impl PartialEq for super::Foo {
        fn eq(&self, _: &super::Foo) -> bool {
            true
        }
    }
}
