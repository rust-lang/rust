pub fn foo() {}

#[macro_export]
macro_rules! macro_2015 {
    () => {
        use edition_lint_paths as other_name;
        use edition_lint_paths::foo as other_foo;
        fn check_macro_2015() {
            ::edition_lint_paths::foo();
        }
    }
}
