//@ edition: 2024
//@ compile-flags: -Znext-solver -Zdxf
//@ check-pass

// This should not break anything.

#![allow(dead_code)]

async fn yield_point() {}

async fn simple_send() {
    let x: i32 = 42;
    yield_point().await;
    let _ = x;
}

async fn with_ref(data: &i32) -> i32 {
    yield_point().await;
    *data
}

fn assert_send(_: impl Send) {}

fn main() {
    assert_send(simple_send());
}
