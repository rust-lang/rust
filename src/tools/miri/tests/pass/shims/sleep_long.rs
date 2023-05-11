//@compile-flags: -Zmiri-ignore-leaks -Zmiri-disable-isolation
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

fn main() {
    let finished = Arc::new(Mutex::new(false));
    let t_finished = finished.clone();
    thread::spawn(move || {
        // Sleep very, very long.
        thread::sleep(Duration::new(u64::MAX, 0));
        *t_finished.lock().unwrap() = true;
    });
    thread::sleep(Duration::from_millis(100));
    assert_eq!(*finished.lock().unwrap(), false);
    // Stopping the main thread will also kill the sleeper.
}
