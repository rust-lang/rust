// Rust briefly used to allow blocks with divergent statements to type check as `!`, even if they
// had a tail expression. This led to a number of regressions (because type information no longer
// flowed from the tail expression) and was quickly reverted.i
//
// See <https://github.com/rust-lang/rust/pull/39485>,
// <https://github.com/rust-lang/rust/pull/40636>,
// <https://github.com/rust-lang/rust/pull/39808>.
//
//@ edition:2015..2021

fn g() {
    &panic!() //~ ERROR mismatched types
}

// This used to ICE, see <https://github.com/rust-lang/rust/issues/10176>
fn f() -> isize {
    (return 1, return 2) //~ ERROR mismatched types
}

fn main() {}
