// edition:2018
#![feature(must_not_suspend)]
#![allow(must_not_suspend)]

use std::future::Future;
use std::sync::Mutex;

fn fake_spawn<F: Future + Send + 'static>(f: F) { }

async fn wrong_mutex() {
  let m = Mutex::new(1);
  {
    let mut guard = m.lock().unwrap();
    (async { "right"; }).await;
    *guard += 1;
  }

  (async { "wrong"; }).await;
}

fn main() {
  fake_spawn(wrong_mutex()); //~ Error future cannot be sent between threads safely
}
