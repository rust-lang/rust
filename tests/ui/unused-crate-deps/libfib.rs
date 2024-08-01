// Test warnings for a library crate

//@ check-pass
//@ aux-crate:bar=bar.rs
//@ compile-flags:--crate-type lib -Wunused-crate-dependencies

pub fn fib(n: u32) -> Vec<u32> {
//~^ WARNING extern crate `bar` is unused in
let mut prev = 0;
    let mut cur = 1;
    let mut v = vec![];

    for _ in 0..n {
        v.push(prev);
        let n = prev + cur;
        prev = cur;
        cur = n;
    }

    v
}
