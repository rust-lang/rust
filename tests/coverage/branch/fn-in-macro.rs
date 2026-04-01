#![feature(coverage_attribute)]
//@ edition: 2024
//@ compile-flags: -Zcoverage-options=branch
//@ llvm-cov-flags: --show-branches=count

// Snapshot test demonstrating how branch coverage interacts with code in macros.
// This test captures current behavior, which is not necessarily "correct".

macro_rules! define_fn {
    () => {
        /// Function defined entirely within a macro.
        fn fn_in_macro() {
            if core::hint::black_box(true) {
                say("true");
            } else {
                say("false");
            }
        }
    };
}

define_fn!();

/// Function not in a macro at all, for comparison.
fn fn_not_in_macro() {
    if core::hint::black_box(true) {
        say("true");
    } else {
        say("false");
    }
}

/// Function that is not in a macro, containing a branch that is in a macro.
fn branch_in_macro() {
    macro_rules! macro_with_branch {
        () => {{
            if core::hint::black_box(true) {
                say("true");
            } else {
                say("false");
            }
        }};
    }

    macro_with_branch!();
}

#[coverage(off)]
fn main() {
    fn_in_macro();
    fn_not_in_macro();
    branch_in_macro();
}

#[coverage(off)]
fn say(message: &str) {
    println!("{message}");
}
