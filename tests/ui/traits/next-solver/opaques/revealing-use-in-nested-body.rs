// Regression test for #137751. This previously ICE'd as
// we did not provide the hidden type of the opaque inside
// of the async block. This caused borrowck of the recursive
// call to ICE.

//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ edition: 2021
//@ check-pass
async fn test() {
    Box::pin(test()).await;
}
fn main() {}
