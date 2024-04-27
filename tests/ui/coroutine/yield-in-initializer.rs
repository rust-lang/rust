//@ run-pass

#![feature(coroutines)]

fn main() {
    #[coroutine] static || { //~ WARN unused coroutine that must be used
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
