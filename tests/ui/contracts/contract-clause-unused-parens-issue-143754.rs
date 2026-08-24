//@ check-pass
// Regression test for <https://github.com/rust-lang/rust/issues/143754>.
// The contract macros wrap the clause in braces rather than parentheses, so `unused_parens`
// must not fire on a contract attribute (and must not emit the attribute-eating suggestion).

#![expect(incomplete_features)]
#![feature(contracts)]
#![deny(unused_parens)]

#[core::contracts::requires(x.baz > 0)]
#[core::contracts::ensures(|ret| *ret > 100)]
fn nest(x: Baz) -> i32 {
    loop {
        return x.baz + 50;
    }
}

struct Baz {
    baz: i32,
}

fn main() {}
