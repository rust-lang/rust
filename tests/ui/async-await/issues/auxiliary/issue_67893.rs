// edition:2018

use std::sync::{Arc, Mutex};

pub async fn f(_: ()) {}

pub async fn run() {
    let x: Arc<Mutex<()>> = unimplemented!();
    f(*x.lock().unwrap()).await;
}
