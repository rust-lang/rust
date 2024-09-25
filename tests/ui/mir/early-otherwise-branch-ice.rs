// Changes in https://github.com/rust-lang/rust/pull/129047 lead to several mir-opt ICE regressions,
// this test is added to make sure this does not regress.

//@ compile-flags: -C opt-level=3
//@ check-pass

#![crate_type = "lib"]

use std::task::Poll;

pub fn poll(val: Poll<Result<Option<Vec<u8>>, u8>>) {
    match val {
        Poll::Ready(Ok(Some(_trailers))) => {}
        Poll::Ready(Err(_err)) => {}
        Poll::Ready(Ok(None)) => {}
        Poll::Pending => {}
    }
}
