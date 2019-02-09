// run-pass

#![feature(generators)]

fn main() {
    static || {
        loop {
            // Test that `opt` is not live across the yield, even when borrowed in a loop.
            // See issue #52792.
            let opt = {
                yield;
                true
            };
            &opt;
        }
    };
}
