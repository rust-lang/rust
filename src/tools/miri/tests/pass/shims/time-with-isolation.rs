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
    // The virtual clock is deterministic and I got 15ms on a 64-bit Linux machine. However, this
    // changes according to the platform so we use an interval to be safe. This should be updated
    // if `NANOSECONDS_PER_BASIC_BLOCK` changes. It may also need updating if the standard library
    // code that runs in the loop above changes.
    assert!(diff.as_millis() > 5);
    assert!(diff.as_millis() < 20);
}

fn test_block_for_one_second() {
    let end = Instant::now() + Duration::from_secs(1);
    // This takes a long time, as we only increment when statements are executed.
    // When we sleep on all threads, we fast forward to the sleep duration, which we can't
    // do with busy waiting.
    while Instant::now() < end {}
}

/// Ensures that we get the same behavior across all targets.
fn test_deterministic() {
    let begin = Instant::now();
    for _ in 0..10_000 {}
    let time = begin.elapsed();
    println!("The loop took around {}ms", time.as_millis());
    println!("(It's fine for this number to change when you `--bless` this test.)")
}

fn main() {
    test_time_passes();
    test_block_for_one_second();
    test_sleep();
    test_deterministic();
}
