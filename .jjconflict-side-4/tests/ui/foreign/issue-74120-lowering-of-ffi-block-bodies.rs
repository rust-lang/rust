// Previously this ICE'd because `fn g()` would be lowered, but the block associated with `fn f()`
// wasn't.

//@ compile-flags: --crate-type=lib

extern "C" {
    fn f() {
    //~^ ERROR incorrect function inside `extern` block
        fn g() {}
    }
}
