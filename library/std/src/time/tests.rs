use super::{Duration, Instant, SystemTime, UNIX_EPOCH};

macro_rules! assert_almost_eq {
    ($a:expr, $b:expr) => {{
        let (a, b) = ($a, $b);
        if a != b {
            let (a, b) = if a > b { (a, b) } else { (b, a) };
            assert!(a - Duration::from_micros(1) <= b, "{:?} is not almost equal to {:?}", a, b);
        }
    }};
}

#[test]
fn instant_monotonic() {
    let a = Instant::now();
    loop {
        let b = Instant::now();
        assert!(b >= a);
        if b > a {
            break;
        }
    }
}

#[test]
fn instant_monotonic_concurrent() -> crate::thread::Result<()> {
    let threads: Vec<_> = (0..8)
        .map(|_| {
            crate::thread::spawn(|| {
                let mut old = Instant::now();
                for _ in 0..5_000_000 {
                    let new = Instant::now();
                    assert!(new >= old);
                    old = new;
                }
            })
        })
        .collect();
    for t in threads {
        t.join()?;
    }
    Ok(())
}

#[test]
fn instant_elapsed() {
    let a = Instant::now();
    a.elapsed();
}

#[test]
fn instant_math() {
    let a = Instant::now();
    let b = Instant::now();
    println!("a: {:?}", a);
    println!("b: {:?}", b);
    let dur = b.duration_since(a);
    println!("dur: {:?}", dur);
    assert_almost_eq!(b - dur, a);
    assert_almost_eq!(a + dur, b);

    let second = Duration::SECOND;
    assert_almost_eq!(a - second + second, a);
    assert_almost_eq!(a.checked_sub(second).unwrap().checked_add(second).unwrap(), a);

    // checked_add_duration will not panic on overflow
    let mut maybe_t = Some(Instant::now());
    let max_duration = Duration::from_secs(u64::MAX);
    // in case `Instant` can store `>= now + max_duration`.
    for _ in 0..2 {
        maybe_t = maybe_t.and_then(|t| t.checked_add(max_duration));
    }
    assert_eq!(maybe_t, None);

    // checked_add_duration calculates the right time and will work for another year
    let year = Duration::from_secs(60 * 60 * 24 * 365);
    assert_eq!(a + year, a.checked_add(year).unwrap());
}

#[test]
fn instant_math_is_associative() {
    let now = Instant::now();
    let offset = Duration::from_millis(5);
    // Changing the order of instant math shouldn't change the results,
    // especially when the expression reduces to X + identity.
    assert_eq!((now + offset) - now, (now - now) + offset);
}

#[test]
#[should_panic]
fn instant_duration_since_panic() {
    let a = Instant::now();
    (a - Duration::SECOND).duration_since(a);
}

#[test]
fn instant_checked_duration_since_nopanic() {
    let now = Instant::now();
    let earlier = now - Duration::SECOND;
    let later = now + Duration::SECOND;
    assert_eq!(earlier.checked_duration_since(now), None);
    assert_eq!(later.checked_duration_since(now), Some(Duration::SECOND));
    assert_eq!(now.checked_duration_since(now), Some(Duration::ZERO));
}

#[test]
fn instant_saturating_duration_since_nopanic() {
    let a = Instant::now();
    let ret = (a - Duration::SECOND).saturating_duration_since(a);
    assert_eq!(ret, Duration::ZERO);
}

#[test]
fn system_time_math() {
    let a = SystemTime::now();
    let b = SystemTime::now();
    match b.duration_since(a) {
        Ok(Duration::ZERO) => {
            assert_almost_eq!(a, b);
        }
        Ok(dur) => {
            assert!(b > a);
            assert_almost_eq!(b - dur, a);
            assert_almost_eq!(a + dur, b);
        }
        Err(dur) => {
            let dur = dur.duration();
            assert!(a > b);
            assert_almost_eq!(b + dur, a);
            assert_almost_eq!(a - dur, b);
        }
    }

    let second = Duration::SECOND;
    assert_almost_eq!(a.duration_since(a - second).unwrap(), second);
    assert_almost_eq!(a.duration_since(a + second).unwrap_err().duration(), second);

    assert_almost_eq!(a - second + second, a);
    assert_almost_eq!(a.checked_sub(second).unwrap().checked_add(second).unwrap(), a);

    let one_second_from_epoch = UNIX_EPOCH + Duration::SECOND;
    let one_second_from_epoch2 =
        UNIX_EPOCH + Duration::from_millis(500) + Duration::from_millis(500);
    assert_eq!(one_second_from_epoch, one_second_from_epoch2);

    // checked_add_duration will not panic on overflow
    let mut maybe_t = Some(SystemTime::UNIX_EPOCH);
    let max_duration = Duration::from_secs(u64::MAX);
    // in case `SystemTime` can store `>= UNIX_EPOCH + max_duration`.
    for _ in 0..2 {
        maybe_t = maybe_t.and_then(|t| t.checked_add(max_duration));
    }
    assert_eq!(maybe_t, None);

    // checked_add_duration calculates the right time and will work for another year
    let year = Duration::from_secs(60 * 60 * 24 * 365);
    assert_eq!(a + year, a.checked_add(year).unwrap());
}

#[test]
fn system_time_elapsed() {
    let a = SystemTime::now();
    drop(a.elapsed());
}

#[test]
fn since_epoch() {
    let ts = SystemTime::now();
    let a = ts.duration_since(UNIX_EPOCH + Duration::SECOND).unwrap();
    let b = ts.duration_since(UNIX_EPOCH).unwrap();
    assert!(b > a);
    assert_eq!(b - a, Duration::SECOND);

    let thirty_years = Duration::SECOND * 60 * 60 * 24 * 365 * 30;

    // Right now for CI this test is run in an emulator, and apparently the
    // aarch64 emulator's sense of time is that we're still living in the
    // 70s. This is also true for riscv (also qemu)
    //
    // Otherwise let's assume that we're all running computers later than
    // 2000.
    if !cfg!(target_arch = "aarch64") && !cfg!(target_arch = "riscv64") {
        assert!(a > thirty_years);
    }

    // let's assume that we're all running computers earlier than 2090.
    // Should give us ~70 years to fix this!
    let hundred_twenty_years = thirty_years * 4;
    assert!(a < hundred_twenty_years);
}
