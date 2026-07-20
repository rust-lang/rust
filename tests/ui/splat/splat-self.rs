//@ run-pass
//@ check-run-results
//! Test using `#[splat]` on self arguments of trait methods.

#![feature(splat)]
#![expect(incomplete_features)]

trait Trait {
    fn method(#[splat] self: Self);
}

impl Trait for (i32, i64) {
    fn method(#[splat] self: Self) {
        println!("{self:?}");
    }
}

fn main() {
    (1_i32, 2_i64).method();
    Trait::method(3_i32, 4_i64);
}
