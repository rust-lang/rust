// compile-flags: -Zmiri-disable-isolation

use std::time::{SystemTime, Instant};

fn main() {
    // Check `SystemTime`.
    let now1 = SystemTime::now();
    // Do some work to make time pass.
    for _ in 0..10 { drop(vec![42]); }
    let now2 = SystemTime::now();
    assert!(now2 > now1);
    let diff = now2.duration_since(now1).unwrap();
    assert_eq!(now1 + diff, now2);
    assert_eq!(now2 - diff, now1);
    // Sanity-check the time we got.
    let seconds_since_epoch = now1.duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs();
    let years_since_epoch = seconds_since_epoch / 3600 / 24 / 365;
    let year = 1970 + years_since_epoch;
    assert!(2020 <= year && year < 2100);

    // Check `Instant`.
    let now1 = Instant::now();
    // Do some work to make time pass.
    for _ in 0..10 { drop(vec![42]); }
    let now2 = Instant::now();
    assert!(now2 > now1);

    #[cfg(not(target_os = "macos"))] // TODO: macOS does not support Instant subtraction
    {
        let diff = now2.duration_since(now1);
        assert_eq!(now1 + diff, now2);
        assert_eq!(now2 - diff, now1);
        // Sanity-check the difference we got.
        assert!(diff.as_micros() > 1);
        assert!(diff.as_micros() < 1_000_000);
    }
}
