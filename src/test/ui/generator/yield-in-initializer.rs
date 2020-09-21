// run-pass

#![feature(generators)]

fn main() {
    static || { //~ WARN unused generator that must be used
        loop {
            // Test that `opt` is not live across the yield, even when borrowed in a loop
            // See https://github.com/rust-lang/rust/issues/52792
            let opt = {
                yield;
                true
            };
            let _ = &opt;
        }
    };
}
