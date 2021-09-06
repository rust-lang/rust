//! This tests the `Integer::{log,log2,log10}` methods. These tests are in a
//! separate file because there's both a large number of them, and not all tests
//! can be run on Android. This is because in Android `log2` uses an imprecise
//! approximation:https://github.com/rust-lang/rust/blob/4825e12fc9c79954aa0fe18f5521efa6c19c7539/src/libstd/sys/unix/android.rs#L27-L53

#[test]
fn checked_log() {
    assert_eq!(999u32.checked_log(10), Some(2));
    assert_eq!(1000u32.checked_log(10), Some(3));
    assert_eq!(555u32.checked_log(13), Some(2));
    assert_eq!(63u32.checked_log(4), Some(2));
    assert_eq!(64u32.checked_log(4), Some(3));
    assert_eq!(10460353203u64.checked_log(3), Some(21));
    assert_eq!(10460353202u64.checked_log(3), Some(20));
    assert_eq!(147808829414345923316083210206383297601u128.checked_log(3), Some(80));
    assert_eq!(147808829414345923316083210206383297600u128.checked_log(3), Some(79));
    assert_eq!(22528399544939174411840147874772641u128.checked_log(19683), Some(8));
    assert_eq!(22528399544939174411840147874772631i128.checked_log(19683), Some(7));

    assert_eq!(0u8.checked_log(4), None);
    assert_eq!(0u16.checked_log(4), None);
    assert_eq!(0i8.checked_log(4), None);
    assert_eq!(0i16.checked_log(4), None);

    for i in i16::MIN..=0 {
        assert_eq!(i.checked_log(4), None);
    }
    for i in 1..=i16::MAX {
        assert_eq!(i.checked_log(13), Some((i as f32).log(13.0) as u32));
    }
    for i in 1..=u16::MAX {
        assert_eq!(i.checked_log(13), Some((i as f32).log(13.0) as u32));
    }
}

#[test]
fn checked_log2() {
    assert_eq!(5u32.checked_log2(), Some(2));
    assert_eq!(0u64.checked_log2(), None);
    assert_eq!(128i32.checked_log2(), Some(7));
    assert_eq!((-55i16).checked_log2(), None);

    assert_eq!(0u8.checked_log2(), None);
    assert_eq!(0u16.checked_log2(), None);
    assert_eq!(0i8.checked_log2(), None);
    assert_eq!(0i16.checked_log2(), None);

    for i in 1..=u8::MAX {
        assert_eq!(i.checked_log2(), Some((i as f32).log2() as u32));
    }
    for i in 1..=u16::MAX {
        // Guard against Android's imprecise f32::log2 implementation.
        if i != 8192 && i != 32768 {
            assert_eq!(i.checked_log2(), Some((i as f32).log2() as u32));
        }
    }
    for i in i8::MIN..=0 {
        assert_eq!(i.checked_log2(), None);
    }
    for i in 1..=i8::MAX {
        assert_eq!(i.checked_log2(), Some((i as f32).log2() as u32));
    }
    for i in i16::MIN..=0 {
        assert_eq!(i.checked_log2(), None);
    }
    for i in 1..=i16::MAX {
        // Guard against Android's imprecise f32::log2 implementation.
        if i != 8192 {
            assert_eq!(i.checked_log2(), Some((i as f32).log2() as u32));
        }
    }
}

// Validate cases that fail on Android's imprecise float log2 implementation.
#[test]
#[cfg(not(target_os = "android"))]
fn checked_log2_not_android() {
    assert_eq!(8192u16.checked_log2(), Some((8192f32).log2() as u32));
    assert_eq!(32768u16.checked_log2(), Some((32768f32).log2() as u32));
    assert_eq!(8192i16.checked_log2(), Some((8192f32).log2() as u32));
}

#[test]
fn checked_log10() {
    assert_eq!(0u8.checked_log10(), None);
    assert_eq!(0u16.checked_log10(), None);
    assert_eq!(0i8.checked_log10(), None);
    assert_eq!(0i16.checked_log10(), None);

    for i in i16::MIN..=0 {
        assert_eq!(i.checked_log10(), None);
    }
    for i in 1..=i16::MAX {
        assert_eq!(i.checked_log10(), Some((i as f32).log10() as u32));
    }
    for i in 1..=u16::MAX {
        assert_eq!(i.checked_log10(), Some((i as f32).log10() as u32));
    }
}

macro_rules! log10_loop {
    ($T:ty, $log10_max:expr) => {
        assert_eq!(<$T>::MAX.log10(), $log10_max);
        for i in 0..=$log10_max {
            let p = (10 as $T).pow(i as u32);
            if p >= 10 {
                assert_eq!((p - 9).log10(), i - 1);
                assert_eq!((p - 1).log10(), i - 1);
            }
            assert_eq!(p.log10(), i);
            assert_eq!((p + 1).log10(), i);
            if p >= 10 {
                assert_eq!((p + 9).log10(), i);
            }

            // also check `x.log(10)`
            if p >= 10 {
                assert_eq!((p - 9).log(10), i - 1);
                assert_eq!((p - 1).log(10), i - 1);
            }
            assert_eq!(p.log(10), i);
            assert_eq!((p + 1).log(10), i);
            if p >= 10 {
                assert_eq!((p + 9).log(10), i);
            }
        }
    };
}

#[test]
fn log10_u8() {
    log10_loop! { u8, 2 }
}

#[test]
fn log10_u16() {
    log10_loop! { u16, 4 }
}

#[test]
fn log10_u32() {
    log10_loop! { u32, 9 }
}

#[test]
fn log10_u64() {
    log10_loop! { u64, 19 }
}

#[test]
fn log10_u128() {
    log10_loop! { u128, 38 }
}
