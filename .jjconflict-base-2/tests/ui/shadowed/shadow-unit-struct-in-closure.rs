//! Regression test for <https://github.com/rust-lang/rust/issues/33504>.
//! Test shadowing a unit-like struct in a closure doesn't cause ICE.

struct Test;

fn main() {
    || {
        let Test = 1; //~ ERROR mismatched types
    };
}
