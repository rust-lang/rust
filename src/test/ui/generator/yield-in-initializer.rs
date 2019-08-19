// run-pass

#![feature(generators)]

fn main() {
    static || {
        loop {
            // Test that `opt` is not live across the yield, even when borrowed in a loop
            // See https://github.com/rust-lang/rust/issues/52792
            let opt = {
                yield;
                true
            };
            &opt;
        }
    };
}
