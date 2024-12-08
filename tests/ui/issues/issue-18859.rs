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
    assert_eq!(module_path!(), "issue_18859");
    assert_eq!(foo::bar::baz::name(), "issue_18859::foo::bar::baz");
}
