// revisions: no_drop_tracking drop_tracking drop_tracking_mir
// [drop_tracking] compile-flags: -Zdrop-tracking
// [drop_tracking_mir] compile-flags: -Zdrop-tracking-mir
// edition:2018

use std::sync::{Arc, Mutex};

pub async fn f(_: ()) {}

pub async fn run() {
    let x: Arc<Mutex<()>> = unimplemented!();
    f(*x.lock().unwrap()).await;
}
