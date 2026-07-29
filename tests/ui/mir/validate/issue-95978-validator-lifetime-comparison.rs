//@ check-pass
//@ compile-flags: -Zvalidate-mir

#![allow(unpredictable_function_pointer_comparisons)]

fn foo(_a: &str) {}

fn main() {
    let x = foo as fn(&'static str);

    let _ = x == foo;
}
