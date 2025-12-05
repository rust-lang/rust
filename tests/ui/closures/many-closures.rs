//! Test that the compiler can handle code bases with a high number of closures.
//! This is particularly important for the MinGW toolchain which has a limit of
//! 2^15 weak symbols per binary. This test creates 2^12 closures (256 functions
//! with 16 closures each) to check the compiler handles this correctly.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/34793>.
//! See also <https://github.com/rust-lang/rust/pull/34830>.

//@ run-pass

// Make sure we don't optimize anything away:
//@ compile-flags: -C no-prepopulate-passes -Cpasses=name-anon-globals

/// Macro for exponential expansion - creates 2^n copies of the given macro call
macro_rules! go_bacterial {
    ($mac:ident) => ($mac!());
    ($mac:ident 1 $($t:tt)*) => (
        go_bacterial!($mac $($t)*);
        go_bacterial!($mac $($t)*);
    )
}

/// Creates and immediately calls a closure
macro_rules! create_closure {
    () => {
        (move || {})()
    };
}

/// Creates a function containing 16 closures (2^4)
macro_rules! create_function_with_closures {
    () => {
        {
            fn function_with_closures() {
                // Create 16 closures using exponential expansion: 2^4 = 16
                go_bacterial!(create_closure 1 1 1 1);
            }
            let _ = function_with_closures();
        }
    }
}

fn main() {
    // Create 2^8 = 256 functions, each containing 16 closures,
    // resulting in 2^12 = 4096 closures total.
    go_bacterial!(create_function_with_closures 1 1 1 1  1 1 1 1);
}
