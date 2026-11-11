//@ check-pass
//@ compile-flags: -Znext-solver

// Ensure that a stack of coerce predicates doesn't end up overflowing when they get procesed
// in *reverse* order, which may require O(N) iterations of the fulfillment loop.

#![recursion_limit = "16"]

fn main() {
    match 0 {
        0 => None,
        1 => None,
        2 => None,
        3 => None,
        4 => None,
        5 => None,
        6 => None,
        7 => None,
        8 => None,
        9 => None,
        10 => None,
        11 => None,
        12 => None,
        13 => None,
        14 => None,
        15 => None,
        16 => None,
        17 => None,
        _ => Some(1u32),
    };
}
