//@ check-pass

// This test makes sure we always considers `const _` items as live for dead code analysis.

#![deny(dead_code)]

const fn is_nonzero(x: u8) -> bool {
    x != 0
}

const _: () = {
    assert!(is_nonzero(2));
};

fn main() {}
