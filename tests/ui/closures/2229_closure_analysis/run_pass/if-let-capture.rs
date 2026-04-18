//@ edition:2021
//@ run-pass

// Regression test for #153982: `if let` in a closure should capture only the
// moved field, matching `match` and plain `let` destructuring.

#![allow(dead_code, irrefutable_let_patterns)]

use std::mem::{size_of, size_of_val};

struct Thing(String, String);

fn if_let_capture_size(x: Thing) -> usize {
    let closure = || {
        if let Thing(_a, _) = x {}
    };

    size_of_val(&closure)
}

fn match_capture_size(x: Thing) -> usize {
    let closure = || {
        match x {
            Thing(_a, _) => {}
        }
    };

    size_of_val(&closure)
}

fn let_capture_size(x: Thing) -> usize {
    let closure = || {
        let Thing(_a, _) = x;
    };

    size_of_val(&closure)
}

fn main() {
    let if_let_size = if_let_capture_size(Thing(String::from("a"), String::from("b")));
    let match_size = match_capture_size(Thing(String::from("a"), String::from("b")));
    let let_size = let_capture_size(Thing(String::from("a"), String::from("b")));

    assert_eq!(if_let_size, size_of::<String>());
    assert_eq!(match_size, size_of::<String>());
    assert_eq!(let_size, size_of::<String>());

    assert_eq!(if_let_size, match_size);
    assert_eq!(if_let_size, let_size);

    // The closure should capture only the moved field, so its size should be
    // less than the size of `Thing`, which would indicate that it captures the
    // entire struct.
    assert!(if_let_size <= size_of::<Thing>());
}
