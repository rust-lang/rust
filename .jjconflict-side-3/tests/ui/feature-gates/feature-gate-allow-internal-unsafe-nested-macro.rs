// gate-test-allow_internal_unsafe

#![allow(unused_macros)]

macro_rules! bar {
    () => {
        // more layers don't help:
        #[allow_internal_unsafe] //~ ERROR the `allow_internal_unsafe` attribute side-steps
        macro_rules! baz {
            () => {}
        }
    }
}

bar!();

fn main() {}
