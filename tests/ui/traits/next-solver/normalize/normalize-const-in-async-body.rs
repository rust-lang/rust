//@ compile-flags: -Znext-solver
//@ check-pass
//@ edition:2021

// Regression test for https://github.com/rust-lang/rust/issues/129865.

pub async fn cleanse_old_array_async(_: &[u8; BUCKET_LEN]) {}

pub const BUCKET_LEN: usize = 0;

pub fn cleanse_old_array_async2() -> impl std::future::Future {
    let x: [u8; 0 + 0] = [];
    async move { let _ = x; }
}

fn main() {}
