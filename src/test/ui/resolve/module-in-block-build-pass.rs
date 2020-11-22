// build-pass

const CONST: &str = "OUTER";
fn bar() -> &'static str { "outer" }

fn module_in_function_use_super() {
    mod inner {
        use super::{bar, CONST};
        fn call_bar() {
            bar();
        }

        fn get_const() -> &'static str {
            CONST
        }
    }
}

fn module_in_function_resolve_super() {
    mod inner {
        fn call_bar() {
            super::bar();
        }

        fn get_const() -> &'static str {
            super::CONST
        }
    }
}


fn module_in_function_use_super_glob() {
    mod inner {
        use super::*;
        fn call_bar() {
            bar();
        }

        fn get_const() -> &'static str {
            CONST
        }
    }
}

fn module_in_block_use_super() {
    {
        mod inner {
            use super::{bar, CONST};
            fn call_bar() {
                bar();
            }

            fn get_const() -> &'static str {
                CONST
            }
        }
    }
}

fn module_in_block_resolve_super() {
    {
        mod inner {
            fn call_bar() {
                super::bar();
            }

            fn get_const() -> &'static str {
                super::CONST
            }
        }
    }
}


fn module_in_block_use_super_glob() {
    {
        mod inner {
            use super::*;
            fn call_bar() {
                bar();
            }

            fn get_const() -> &'static str {
                CONST
            }
        }
    }
}

fn main() {
    module_in_function_use_super();
    module_in_function_resolve_super();
    module_in_function_use_super_glob();

    module_in_block_use_super();
    module_in_block_resolve_super();
    module_in_block_use_super_glob();
}
