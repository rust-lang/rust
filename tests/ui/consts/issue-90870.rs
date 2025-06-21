// Regression test for issue #90870.
//@ check-pass
#![feature(const_trait_impl)]
#![allow(dead_code)]

const fn f(a: &u8, b: &u8) -> bool {
    a == b
}

const fn g(a: &&&&i64, b: &&&&i64) -> bool {
    a == b
}

const fn h(mut a: &[u8], mut b: &[u8]) -> bool {
    while let ([l, at @ ..], [r, bt @ ..]) = (a, b) {
        if l == r {
            a = at;
            b = bt;
        } else {
            return false;
        }
    }

    a.is_empty() && b.is_empty()
}

fn main() {}
