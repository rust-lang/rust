// Regression test for #154800 and #154847
//@ compile-flags: --crate-type=lib
//@ check-pass

#[expect]
#[cfg(false)]
fn main() {
    let _ : fn(#[expect[]] i32);
}
