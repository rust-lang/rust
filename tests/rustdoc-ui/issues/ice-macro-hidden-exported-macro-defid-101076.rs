//@ check-pass
// https://github.com/rust-lang/rust/issues/101076

const _: () = {
    #[macro_export]
    macro_rules! first_macro {
        () => {}
    }
    mod foo {
        #[macro_export]
        macro_rules! second_macro {
            () => {}
        }
    }
};
