//@ compile-flags: -Znext-solver
//@ check-pass

// When canonicalizing responses, we bail if there are too many inference variables.
// We previously also counted placeholders, which is incorrect.
#![recursion_limit = "8"]

fn foo<T>() {}

fn bar<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>() {
    // The query response will contain 10 placeholders, which previously
    // caused us to bail here.
    foo::<(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9)>();
}

fn main() {}
