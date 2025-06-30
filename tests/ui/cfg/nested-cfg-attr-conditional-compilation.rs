//! Test that nested `cfg_attr` attributes work correctly for conditional compilation.
//! This checks that `cfg_attr` can be arbitrarily deeply nested and that the
//! expansion works from outside to inside, eventually applying the innermost
//! conditional compilation directive.
//!
//! In this test, `cfg_attr(all(), cfg_attr(all(), cfg(false)))` should expand to:
//! 1. `cfg_attr(all(), cfg(false))` (outer cfg_attr applied)
//! 2. `cfg(false)` (inner cfg_attr applied)
//! 3. Function `f` is excluded from compilation
//!
//! Added in <https://github.com/rust-lang/rust/pull/34216>.

#[cfg_attr(all(), cfg_attr(all(), cfg(false)))]
fn f() {}

fn main() {
    f() //~ ERROR cannot find function `f` in this scope
}
