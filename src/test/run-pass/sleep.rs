// ignore-emscripten no threads support
// ignore-sgx no thread sleep support

use std::thread::{self, sleep};
use std::time::Duration;
use std::sync::{Arc, Mutex};
use std::u64;

fn main() {
    let finished = Arc::new(Mutex::new(false));
    let t_finished = finished.clone();
    thread::spawn(move || {
        sleep(Duration::new(u64::MAX, 0));
        *t_finished.lock().unwrap() = true;
    });
    sleep(Duration::from_millis(100));
    assert_eq!(*finished.lock().unwrap(), false);
}
