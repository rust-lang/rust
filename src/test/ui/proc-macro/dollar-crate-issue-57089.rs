// compile-pass
// edition:2018
// aux-build:dollar-crate.rs

// Anonymize unstable non-dummy spans while still showing dummy spans `0..0`.
// normalize-stdout-test "bytes\([^0]\w*\.\.(\w+)\)" -> "bytes(LO..$1)"
// normalize-stdout-test "bytes\((\w+)\.\.[^0]\w*\)" -> "bytes($1..HI)"

extern crate dollar_crate;

type S = u8;

macro_rules! m {
    () => {
        dollar_crate::m_empty! {
            struct M($crate::S);
        }

        #[dollar_crate::a]
        struct A($crate::S);
    };
}

m!();

fn main() {}
