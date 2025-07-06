//@compile-flags: -Zmiri-disable-isolation
#![feature(thread_sleep_until)]

use std::time::{Duration, Instant, SystemTime};

fn duration_sanity(diff: Duration) {
    // On my laptop, I observed times around 15-40ms. Add 10x lee-way both ways.
    assert!(diff.as_millis() > 1);
    assert!(diff.as_millis() < 1000); // macOS is very slow sometimes
}

fn test_sleep() {
    let before = Instant::now();
    std::thread::sleep(Duration::from_millis(100));
    let after = Instant::now();
    assert!((after - before).as_millis() >= 100);
}

fn test_sleep_until() {
    let before = Instant::now();
    let hunderd_millis_after_start = before + Duration::from_millis(100);
    std::thread::sleep_until(hunderd_millis_after_start);
    let after = Instant::now();
    assert!((after - before).as_millis() >= 100);
}

fn main() {
    // Check `SystemTime`.
    let now1 = SystemTime::now();
    let seconds_since_epoch = now1.duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs();
    let years_since_epoch = seconds_since_epoch / 3600 / 24 / 365;
    let year = 1970 + years_since_epoch;
    assert!(2020 <= year && year < 2100);
    // Do some work to make time pass.
    for _ in 0..10 {
        drop(vec![42]);
    }
    let now2 = SystemTime::now();
    assert!(now2 > now1);
    // Sanity-check the difference we got.
    let diff = now2.duration_since(now1).unwrap();
    assert_eq!(now1 + diff, now2);
    assert_eq!(now2 - diff, now1);
    duration_sanity(diff);

    // Check `Instant`.
    let now1 = Instant::now();
    // Do some work to make time pass.
    for _ in 0..10 {
        drop(vec![42]);
    }
    let now2 = Instant::now();
    assert!(now2 > now1);
    // Sanity-check the difference we got.
    let diff = now2.duration_since(now1);
    assert_eq!(now1 + diff, now2);
    assert_eq!(now2 - diff, now1);
    duration_sanity(diff);

    test_sleep();
    test_sleep_until();
}
