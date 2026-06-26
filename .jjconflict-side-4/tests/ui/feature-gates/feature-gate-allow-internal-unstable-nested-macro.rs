// gate-test-allow_internal_unstable

#![allow(unused_macros)]

macro_rules! bar {
    () => {
        // more layers don't help:
        #[allow_internal_unstable()] //~ ERROR the `allow_internal_unstable` attribute side-steps
        macro_rules! baz {
            () => {}
        }
    }
}

bar!();

fn main() {}
