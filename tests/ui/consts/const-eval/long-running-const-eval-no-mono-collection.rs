// Regression test for <https://github.com/rust-lang/rust/issues/166278>.

//@ compile-flags: -Ztiny-const-eval-limit --emit=link

#![crate_type = "lib"]

pub fn f() {
    [(); loop {}];
    //~^ ERROR is taking a long time
}
