// gate-test-allow_internal_unsafe

#![allow(unused_macros)]

macro_rules! bar {
    () => {
        // more layers don't help:
        #[allow_internal_unsafe] //~ ERROR allow_internal_unsafe side-steps
        macro_rules! baz {
            () => {}
        }
    }
}

bar!();

fn main() {}
