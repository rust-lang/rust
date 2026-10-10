//@ check-pass
//@ revisions: current next
//@[current] compile-flags: -Znext-solver=coherence
//@[next] compile-flags: -Znext-solver=globally

// https://github.com/rust-lang/rust/issues/161495

#![warn(clippy::large_futures)]

async fn callee() {}

async fn caller() {
    callee().await;
    std::pin::pin!(callee()).await;
}

fn main() {}
