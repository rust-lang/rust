// Regression test for #137751. This previously ICE'd as
// we did not provide the hidden type of the opaque inside
// of the async block. This caused borrowck of the recursive
// call to ICE.

//@ compile-flags: --edition=2021
//@ check-pass
async fn test() {
    Box::pin(test()).await;
}
fn main() {}
