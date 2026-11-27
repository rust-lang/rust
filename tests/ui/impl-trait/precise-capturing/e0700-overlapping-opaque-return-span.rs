// Regression test for <https://github.com/rust-lang/rust/issues/164265>.
// The new solver used to point the hidden type at the whole function header.

//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver=globally

fn overlapping_opaque_return_span(x: &i32) -> impl Sized + use<> {
    x
    //[current]~^ ERROR hidden type for `impl Sized` captures lifetime that does not appear in bounds
}
//[next]~^^^^ ERROR hidden type for `impl Sized` captures lifetime that does not appear in bounds

fn main() {}
