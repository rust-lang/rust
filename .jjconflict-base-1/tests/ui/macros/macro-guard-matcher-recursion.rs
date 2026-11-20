//! Regression test for <https://github.com/rust-lang/rust/issues/155333>
#![feature(macro_guard_matcher)]
fn main() {
    macro_rules! m {
        ($g : guard) => {
            m!($g) //~ ERROR recursion limit reached while expanding `m!`
        };
    }
    m!(if x)
}
