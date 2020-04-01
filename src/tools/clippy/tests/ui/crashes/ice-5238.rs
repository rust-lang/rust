// Regression test for #5238 / https://github.com/rust-lang/rust/pull/69562

#![feature(generators, generator_trait)]

fn main() {
    let _ = || {
        yield;
    };
}
