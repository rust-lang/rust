//@ compile-flags: -Znext-solver
//@ edition: 2024
//@ check-pass

// We don't increase recursion depth when proving auto traits for witness
// and rigid opaques. This is to mitigate the FCW warnings in deeply nested
// async calls. See #159228.

#![recursion_limit = "6"]

async fn foo1() {}

async fn foo2() {
    foo1().await
}
async fn foo3() {
    foo2().await
}
async fn foo4() {
    foo3().await
}

async fn foo5() {
    foo4().await
}

fn assert_send<T: Send>(_: T) {}

fn main() {
    assert_send(foo5());
}
