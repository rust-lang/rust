//@ edition:2018

use std::sync::{Arc, Mutex};

fn make_arc() -> Arc<Mutex<()>> { unimplemented!() }

pub async fn f(_: ()) {}

pub async fn run() {
    let x: Arc<Mutex<()>> = make_arc();
    f(*x.lock().unwrap()).await;
}
