#![warn(clippy::same_name_method)]
#![allow(dead_code, non_camel_case_types)]

trait T1 {
    fn foo() {}
}

trait T2 {
    fn foo() {}
}

mod should_lint {

    mod test_basic_case {
        use crate::T1;

        struct S;

        impl S {
            fn foo() {}
            //~^ same_name_method
        }

        impl T1 for S {
            fn foo() {}
        }
    }

    mod test_derive {

        #[derive(Clone)]
        struct S;

        impl S {
            fn clone() {}
            //~^ same_name_method
        }
    }

    mod with_generic {
        use crate::T1;

        struct S<U>(U);

        impl<U> S<U> {
            fn foo() {}
            //~^ same_name_method
        }

        impl<U: Copy> T1 for S<U> {
            fn foo() {}
        }
    }

    mod default_method {
        use crate::T1;

        struct S;

        impl S {
            fn foo() {}
            //~^ same_name_method
        }

        impl T1 for S {}
    }

    mod multiple_conflicting_traits {
        use crate::{T1, T2};

        struct S;

        impl S {
            fn foo() {}
            //~^ same_name_method
            //~| same_name_method
        }

        impl T1 for S {}

        impl T2 for S {}
    }
}

mod should_not_lint {

    mod not_lint_two_trait_method {
        use crate::{T1, T2};

        struct S;

        impl T1 for S {
            fn foo() {}
        }

        impl T2 for S {
            fn foo() {}
        }
    }

    mod only_lint_on_method {
        trait T3 {
            type foo;
        }

        struct S;

        impl S {
            fn foo() {}
        }
        impl T3 for S {
            type foo = usize;
        }
    }
}

mod check_expect_suppression {
    use crate::T1;

    struct S;

    impl S {
        #[expect(clippy::same_name_method)]
        fn foo() {}
    }

    impl T1 for S {
        fn foo() {}
    }
}

fn main() {}
