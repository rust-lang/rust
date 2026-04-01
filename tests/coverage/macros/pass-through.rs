#![feature(coverage_attribute)]
//@ edition: 2024

// Test that when a macro expands to another macro, without any significant
// spans of its own, that this doesn't cause coverage instrumentation to give
// up and ignore the inner spans.

macro_rules! inner_macro {
    () => {
        if core::hint::black_box(true) {
            say("true");
        } else {
            say("false");
        }
    };
}

macro_rules! middle_macro {
    () => {
        inner_macro!()
    };
}

macro_rules! outer_macro {
    () => {
        middle_macro!()
    };
}

// In each of these three functions, the macro call should be instrumented,
// and should have an execution count of 1.
//
// Each function contains some extra code to ensure that control flow is
// non-trivial.

fn uses_inner_macro() {
    if core::hint::black_box(true) {
        say("before inner_macro");
        inner_macro!();
        say("after inner_macro");
    }
}

fn uses_middle_macro() {
    if core::hint::black_box(true) {
        say("before middle_macro");
        middle_macro!();
        say("after middle_macro")
    }
}

fn uses_outer_macro() {
    if core::hint::black_box(true) {
        say("before outer_macro");
        outer_macro!();
        say("after outer_macro");
    }
}

#[coverage(off)]
fn main() {
    uses_inner_macro();
    uses_middle_macro();
    uses_outer_macro();
}

#[coverage(off)]
fn say(message: &str) {
    println!("{message}");
}
