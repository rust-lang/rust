//@aux-build:proc_macros.rs
#![warn(clippy::ifs_as_logical_ops)]
#![expect(clippy::needless_bool)]

extern crate proc_macros;
use proc_macros::{external, with_span};

fn main() {
    // test code goes here
}

fn basic_lint_test(b1: bool, b2: bool) -> bool {
    if b1 { b2 } else { false }
    //~^ ifs_as_logical_ops
}

fn lint_test_with_cfg_as_second_argument(b1: bool, b2: bool) -> bool {
    if b1 { b2 } else { cfg!(false) }
}

fn nested_expressions_produces_lint(b1: bool, b2: bool) -> bool {
    if b1 { { b2 } } else { false }
    //~^ ifs_as_logical_ops
}

fn diverging_does_not_produce_lint(b1: bool, b2: bool) -> bool {
    if b1 { panic!() } else { false }
}

fn complex_expressions_do_not_produce_lint(b1: bool, b2: bool) -> bool {
    if b1 {
        let mut some_value = 100;
        if some_value < 50 {
            return true;
        }
        true
    } else {
        false
    }
}

fn example_with_if_comment_before_expr_does_not_lint(chars: Vec<char>) -> bool {
    if !chars.is_empty() {
        // SAFETY: Always starts with ^[ and ends with m.
        chars.len() > 5
    } else {
        false
    }
}

fn example_with_if_comment_after_expr_does_not_lint(chars: Vec<char>) -> bool {
    if !chars.is_empty() {
        chars.len() > 5
        // SAFETY: Always starts with ^[ and ends with m.
    } else {
        false
    }
}

fn example_with_else_comment_before_expr_does_not_lint(chars: Vec<char>) -> bool {
    if !chars.is_empty() {
        chars.len() > 5
    } else {
        // Watch out! Comment!
        false
    }
}

fn example_with_else_comment_after_expr_does_not_lint(chars: Vec<char>) -> bool {
    if !chars.is_empty() {
        chars.len() > 5
    } else {
        false
        // Watch out! Comment!
    }
}

fn example_with_cfg_does_not_lint(chars: Vec<char>) -> bool {
    if !chars.is_empty() {
        #[cfg(false)]
        return 2 * 2 == 5;
        chars.len() > 5
    } else {
        false
    }
}

fn example_with_debug_does_not_lint(b1: bool) -> bool {
    if b1 {
        dbg!("Something");
        true
    } else {
        false
    }
}

fn basic_if_let_does_not_lint(b1: bool, b2: Option<i32>) -> bool {
    if let Some(30) = b2 { b1 } else { false }
}

fn if_and_let_does_not_lint(b1: bool, b2: Option<i32>, b3: bool) -> bool {
    if b1 && let Some(b) = b2 { b3 } else { false }
}

fn if_let_complex_does_not_lint(b1: bool, b2: Option<i32>, b3: bool) -> bool {
    if let Some(b) = b2
        && b1
    {
        b3
    } else {
        false
    }
}

fn else_if_does_lint(b1: bool, b2: bool, b3: bool) -> bool {
    if b1 {
        // There is some expansion here.
        let _ = 40;
        false
    } else if b2 {
        b3
    } else {
        false
    }
    //~^^^^^ ifs_as_logical_ops
}

fn needless_bool_clash_does_not_lint(x: bool) -> bool {
    if x { true } else { false }
}

macro_rules! always_return_true {
    () => {
        true
    };
}

fn needless_bool_macro_clash_does_not_lint(b1: bool) -> bool {
    if b1 { always_return_true!() } else { false }
}

macro_rules! nothing {
    () => {};
}

fn empty_expansion_does_not_lint(b1: bool, b2: bool) -> bool {
    if b1 {
        nothing!();
        b2
    } else {
        false
    }
}

fn macro_with_comment_does_not_lint(b1: bool, num: i32) -> bool {
    macro_rules! b2 {
        () => {{ num > 33 }};
    }
    if b1 {
        // watch out! Comment!
        b2!()
    } else {
        false
    }
}

fn macro_without_comment_does_lint(b1: bool, num: i32) -> bool {
    macro_rules! b2 {
        () => {{ num > 33 }};
    }
    if b1 { b2!() } else { false }
    //~^ ifs_as_logical_ops
}

fn proc_macro_tests() {
    proc_macros::external! {
        fn would_lint_usually(b1: bool, num: i32) -> bool {
            if b1 { num > 30 } else { false }
        }
    }
    proc_macros::with_span! {
        span
        fn would_lint_usually_2(b1: bool, num: i32) -> bool {
            if b1 { num > 30 } else { false }
        }
    }
}
