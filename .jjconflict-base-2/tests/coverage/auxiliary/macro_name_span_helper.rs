//@ edition: 2021

#[macro_export]
macro_rules! macro_that_defines_a_function {
    (fn $name:ident () $body:tt) => {
        fn $name () -> () $body
    }
}

// Non-executable comment.
