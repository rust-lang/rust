use std::time::{Duration, Instant};

fn test_sleep() {
    // We sleep a *long* time here -- but the clock is virtual so the test should still pass quickly.
    let before = Instant::now();
    std::thread::sleep(Duration::from_secs(3600));
    let after = Instant::now();
    assert!((after - before).as_secs() >= 3600);
}

/// Ensure that time passes even if we don't sleep (but just work).
fn test_time_passes() {
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
    // The virtual clock is deterministic and I got 29us on a 64-bit Linux machine. However, this
    // changes according to the platform so we use an interval to be safe. This should be updated
    // if `NANOSECONDS_PER_BASIC_BLOCK` changes.
    assert!(diff.as_micros() > 10);
    assert!(diff.as_micros() < 40);
}

fn main() {
    test_time_passes();
    test_sleep();
}
