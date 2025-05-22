//! Regression test for ICE #113379. Liveness linting assumes that `continue`s all point to loops.
//! This tests that if a `continue` points to a block, we don't run liveness lints.

async fn f999() -> Vec<usize> {
    //~^ ERROR `async fn` is not permitted in Rust 2015
    'b: {
        //~^ ERROR mismatched types
        continue 'b;
        //~^ ERROR `continue` pointing to a labeled block
    }
}
//~^ ERROR `main` function not found
