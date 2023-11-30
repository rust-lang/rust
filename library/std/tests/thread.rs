#![feature(thread_sleep_until)]

use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

#[test]
#[cfg_attr(target_os = "emscripten", ignore)]
fn sleep() {
    let finished = Arc::new(Mutex::new(false));
    let t_finished = finished.clone();
    thread::spawn(move || {
        thread::sleep(Duration::new(u64::MAX, 0));
        *t_finished.lock().unwrap() = true;
    });
    thread::sleep(Duration::from_millis(100));
    assert_eq!(*finished.lock().unwrap(), false);
}

#[test]
fn sleep_until() {
    let now = Instant::now();
    let period = Duration::from_millis(100);
    let deadline = now + period;
    thread::sleep_until(deadline);

    let margin = Duration::from_micros(100);
    let elapsed = now.elapsed();
    assert!(
        (period..period + margin).contains(&elapsed),
        "elapsed: {} microseconds, expected 100_000 to 100_100",
        elapsed.as_micros()
    );
}
