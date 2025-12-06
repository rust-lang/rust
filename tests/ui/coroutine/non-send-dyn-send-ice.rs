//! Regression test for ICE #137916
//@ edition: 2021
//@ compile-flags: -Zvalidate-mir

use std::ptr::null;

async fn a() -> Box<dyn Send> {
    Box::new(async { //~ ERROR future cannot be sent between threads safely
        let non_send = null::<()>();
        &non_send;
        async {}.await
    })
}

fn main() {}
