// pp-exact
// pretty-compare-only
// edition:2021

async fn f() {
    let first = async { 1 };
    let second = async move { 2 };
    join(first, second).await
}
