// Regression test for issue #90870.

#![allow(dead_code)]

const fn f(a: &u8, b: &u8) -> bool {
    a == b
    //~^ ERROR cannot call conditionally-const method `<&u8 as PartialEq>::eq` in constant functions
}

const fn g(a: &&&&i64, b: &&&&i64) -> bool {
    a == b
    //~^ ERROR cannot call conditionally-const method `<&&&&i64 as PartialEq>::eq` in constant functions
}

const fn h(mut a: &[u8], mut b: &[u8]) -> bool {
    while let ([l, at @ ..], [r, bt @ ..]) = (a, b) {
        if l == r {
        //~^ ERROR cannot call conditionally-const method `<&u8 as PartialEq>::eq` in constant functions
            a = at;
            b = bt;
        } else {
            return false;
        }
    }

    a.is_empty() && b.is_empty()
}

fn main() {}
