// check-pass

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
