macro_rules! sugg_test {
    ( $( $name:ident: $paths:expr => $suggestions:expr ),* ) => {
        $(
            #[test]
            fn $name() {
                let suggestions = crate::get_suggestions(&$paths).into_iter().map(|s| s.to_string()).collect::<Vec<_>>();
                assert_eq!(suggestions, $suggestions);
            }
        )*
    };
}

sugg_test! {
    test_error_code_docs: ["compiler/rustc_error_codes/src/error_codes/E0000.md"] =>
        ["check N/A", "test compiler/rustc_error_codes N/A", "test linkchecker 0", "test tests/ui tests/run-make 1"],

    test_rustdoc: ["src/librustdoc/src/lib.rs"] => ["test rustdoc 1"],

    test_rustdoc_and_libstd: ["src/librustdoc/src/lib.rs", "library/std/src/lib.rs"] =>
        ["test library/std N/A", "test rustdoc 1"]
}
