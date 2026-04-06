// Regression test for #154800 and #154847
//@ compile-flags: --crate-type=lib

#[expect]
//~^ ERROR malformed
#[cfg(false)]
fn main() {
    let _ : fn(#[expect[]] i32);
    //~^ ERROR wrong meta list delimiters
}
