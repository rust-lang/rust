use std::time::{Duration, Instant};

fn duration_sanity(diff: Duration) {
    // The virtual clock is deterministic and I got 29us on a 64-bit Linux machine. However, this
    // changes according to the platform so we use an interval to be safe. This should be updated
    // if `NANOSECONDS_PER_BASIC_BLOCK` changes.
    assert!(diff.as_micros() > 10);
    assert!(diff.as_micros() < 40);
}

fn test_sleep() {
    let before = Instant::now();
    std::thread::sleep(Duration::from_millis(100));
    let after = Instant::now();
    assert!((after - before).as_millis() >= 100);
}

fn main() {
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
}
