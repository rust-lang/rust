//@ known-bug: #137916
//@ edition: 2021
use std::ptr::null;

async fn a() -> Box<dyn Send> {
    Box::new(async {
        let non_send = null::<()>();
        &non_send;
        async {}.await
    })
}

fn main() {}
