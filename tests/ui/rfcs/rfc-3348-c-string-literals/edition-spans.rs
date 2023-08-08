// even if this crate is edition 2021, proc macros compiled using older
// editions should still be able to observe the pre-2021 token behavior
//
// adapted from tests/ui/rust-2021/reserved-prefixes-via-macro.rs

// edition: 2021
// check-pass

// aux-build: count.rs
extern crate count;

const _: () = {
    assert!(count::number_of_tokens!() == 2);
};

fn main() {}
