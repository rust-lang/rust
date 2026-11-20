//! Regression test for <https://github.com/rust-lang/rust/issues/18859>.

//@ run-pass

mod foo {
    pub mod bar {
        pub mod baz {
            pub fn name() -> &'static str {
                module_path!()
            }
        }
    }
}

fn main() {
    assert_eq!(module_path!(), "module_path_in_nested_modules");
    assert_eq!(foo::bar::baz::name(), "module_path_in_nested_modules::foo::bar::baz");
}
