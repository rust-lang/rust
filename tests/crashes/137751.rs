//@ known-bug: #137751
//@ compile-flags: --edition=2021 -Znext-solver=globally
async fn test() {
    Box::pin(test()).await;
}
fn main() {}
