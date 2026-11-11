//@ check-pass
//@ proc-macro: issue-79242.rs

// Regression test for issue #79242
// Tests that compilation time doesn't blow up for a proc-macro
// invocation with deeply nested nonterminals

#![allow(unused)]

extern crate issue_79242;

macro_rules! declare_nats {
    ($prev:ty) => {};
    ($prev:ty, $n:literal$(, $tail:literal)*) => {

        issue_79242::dummy! {
            $prev
        }

        declare_nats!(Option<$prev>$(, $tail)*);
    };
    (0, $($n:literal),+) => {
        pub struct N0;
        declare_nats!(N0, $($n),+);
    };
}

declare_nats! {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28
}


fn main() {}
