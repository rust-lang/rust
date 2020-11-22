// run-pass
// compile-flags:--test

// These two items are marked as allow(dead_code) because none of the
// test code should ever resolve to them.
#[allow(dead_code)]
const CONST: &str = "OUTER";
#[allow(dead_code)]
fn bar() -> &'static str { "outer" }

#[test]
fn module_in_function_prefer_inner() {

    const CONST: &str = "INNER";
    fn bar() -> &'static str { "inner" }

    mod inner {
        use super::{bar, CONST};
        pub fn call_bar() -> &'static str {
            bar()
        }

        pub fn get_const() -> &'static str {
            CONST
        }
    }

    assert_eq!(inner::call_bar(), "inner");
    assert_eq!(inner::get_const(), "INNER")
}

#[test]
fn module_in_function_prefer_inner_glob() {

    const CONST: &str = "INNER";
    fn bar() -> &'static str { "inner" }

    mod inner {
        use super::*;
        pub fn call_bar() -> &'static str {
            bar()
        }

        pub fn get_const() -> &'static str {
            CONST
        }
    }

    assert_eq!(inner::call_bar(), "inner");
    assert_eq!(inner::get_const(), "INNER");
}

#[test]
fn module_in_block_prefer_inner() {

    const CONST: &str = "INNER";

    // anonymous block
    {
        fn bar() -> &'static str { "inner_block" }

        mod inner {
            use super::{CONST, bar};
            pub fn call_bar() -> &'static str {
                bar()
            }

            pub fn get_const() -> &'static str {
                CONST
            }
        }

        assert_eq!(inner::call_bar(), "inner_block");
        assert_eq!(inner::get_const(), "INNER");
    }
}


#[test]
fn module_in_block_prefer_inner_glob() {
    const CONST: &str = "INNER";

    // anonymous block
    {
        fn bar() -> &'static str { "inner_block" }

        mod inner {
            use super::*;
            pub fn call_bar() -> &'static str {
                bar()
            }

            pub fn get_const() -> &'static str {
                CONST
            }
        }

        assert_eq!(inner::call_bar(), "inner_block");
        assert_eq!(inner::get_const(), "INNER");
    }
}
