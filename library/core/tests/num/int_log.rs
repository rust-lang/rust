//! Tests for the `Integer::{ilog,log2,log10}` methods.

#[test]
fn checked_ilog() {
    assert_eq!(999u32.checked_ilog(10), Some(2));
    assert_eq!(1000u32.checked_ilog(10), Some(3));
    assert_eq!(555u32.checked_ilog(13), Some(2));
    assert_eq!(63u32.checked_ilog(4), Some(2));
    assert_eq!(64u32.checked_ilog(4), Some(3));
    assert_eq!(10460353203u64.checked_ilog(3), Some(21));
    assert_eq!(10460353202u64.checked_ilog(3), Some(20));
    assert_eq!(147808829414345923316083210206383297601u128.checked_ilog(3), Some(80));
    assert_eq!(147808829414345923316083210206383297600u128.checked_ilog(3), Some(79));
    assert_eq!(22528399544939174411840147874772641u128.checked_ilog(19683), Some(8));
    assert_eq!(22528399544939174411840147874772631i128.checked_ilog(19683), Some(7));

    assert_eq!(0u8.checked_ilog(4), None);
    assert_eq!(0u16.checked_ilog(4), None);
    assert_eq!(0i8.checked_ilog(4), None);
    assert_eq!(0i16.checked_ilog(4), None);

    #[cfg(not(miri))] // Miri is too slow
    for i in i16::MIN..=0 {
        assert_eq!(i.checked_ilog(4), None, "checking {i}");
    }
    #[cfg(not(miri))] // Miri is too slow
    for i in 1..=i16::MAX {
        assert_eq!(i.checked_ilog(13), Some((i as f32).log(13.0) as u32), "checking {i}");
    }
    #[cfg(not(miri))] // Miri is too slow
    for i in 1..=u16::MAX {
        assert_eq!(i.checked_ilog(13), Some((i as f32).log(13.0) as u32), "checking {i}");
    }
}

#[test]
fn checked_ilog2() {
    assert_eq!(5u32.checked_ilog2(), Some(2));
    assert_eq!(0u64.checked_ilog2(), None);
    assert_eq!(128i32.checked_ilog2(), Some(7));
    assert_eq!((-55i16).checked_ilog2(), None);

    assert_eq!(0u8.checked_ilog2(), None);
    assert_eq!(0u16.checked_ilog2(), None);
    assert_eq!(0i8.checked_ilog2(), None);
    assert_eq!(0i16.checked_ilog2(), None);

    assert_eq!(8192u16.checked_ilog2(), Some((8192f32).log2() as u32));
    assert_eq!(32768u16.checked_ilog2(), Some((32768f32).log2() as u32));
    assert_eq!(8192i16.checked_ilog2(), Some((8192f32).log2() as u32));

    for i in 1..=u8::MAX {
        assert_eq!(i.checked_ilog2(), Some((i as f32).log2() as u32), "checking {i}");
    }
    #[cfg(not(miri))] // Miri is too slow
    for i in 1..=u16::MAX {
        // Guard against Android's imprecise f32::ilog2 implementation.
        if i != 8192 && i != 32768 {
            assert_eq!(i.checked_ilog2(), Some((i as f32).log2() as u32), "checking {i}");
        }
    }
    for i in i8::MIN..=0 {
        assert_eq!(i.checked_ilog2(), None, "checking {i}");
    }
    for i in 1..=i8::MAX {
        assert_eq!(i.checked_ilog2(), Some((i as f32).log2() as u32), "checking {i}");
    }
    #[cfg(not(miri))] // Miri is too slow
    for i in i16::MIN..=0 {
        assert_eq!(i.checked_ilog2(), None, "checking {i}");
    }
    #[cfg(not(miri))] // Miri is too slow
    for i in 1..=i16::MAX {
        // Guard against Android's imprecise f32::ilog2 implementation.
        if i != 8192 {
            assert_eq!(i.checked_ilog2(), Some((i as f32).log2() as u32), "checking {i}");
        }
    }
}

#[test]
fn checked_ilog10() {
    assert_eq!(0u8.checked_ilog10(), None);
    assert_eq!(0u16.checked_ilog10(), None);
    assert_eq!(0i8.checked_ilog10(), None);
    assert_eq!(0i16.checked_ilog10(), None);

    #[cfg(not(miri))] // Miri is too slow
    for i in i16::MIN..=0 {
        assert_eq!(i.checked_ilog10(), None, "checking {i}");
    }
    #[cfg(not(miri))] // Miri is too slow
    for i in 1..=i16::MAX {
        assert_eq!(i.checked_ilog10(), Some((i as f32).log10() as u32), "checking {i}");
    }
    #[cfg(not(miri))] // Miri is too slow
    for i in 1..=u16::MAX {
        assert_eq!(i.checked_ilog10(), Some((i as f32).log10() as u32), "checking {i}");
    }
    #[cfg(not(miri))] // Miri is too slow
    for i in 1..=100_000u32 {
        assert_eq!(i.checked_ilog10(), Some((i as f32).log10() as u32), "checking {i}");
    }
}

macro_rules! ilog10_loop {
    ($T:ty, $ilog10_max:expr) => {
        assert_eq!(<$T>::MAX.ilog10(), $ilog10_max);
        for i in 0..=$ilog10_max {
            let p = (10 as $T).pow(i as u32);
            if p >= 10 {
                assert_eq!((p - 9).ilog10(), i - 1);
                assert_eq!((p - 1).ilog10(), i - 1);
            }
            assert_eq!(p.ilog10(), i);
            assert_eq!((p + 1).ilog10(), i);
            if p >= 10 {
                assert_eq!((p + 9).ilog10(), i);
            }

            // also check `x.ilog(10)`
            if p >= 10 {
                assert_eq!((p - 9).ilog(10), i - 1);
                assert_eq!((p - 1).ilog(10), i - 1);
            }
            assert_eq!(p.ilog(10), i);
            assert_eq!((p + 1).ilog(10), i);
            if p >= 10 {
                assert_eq!((p + 9).ilog(10), i);
            }
        }
    };
}

#[test]
fn ilog10_u8() {
    ilog10_loop! { u8, 2 }
}

#[test]
fn ilog10_u16() {
    ilog10_loop! { u16, 4 }
}

#[test]
fn ilog10_u32() {
    ilog10_loop! { u32, 9 }
}

#[test]
fn ilog10_u64() {
    ilog10_loop! { u64, 19 }
}

#[test]
fn ilog10_u128() {
    ilog10_loop! { u128, 38 }
}

#[test]
#[should_panic(expected = "argument of integer logarithm must be positive")]
fn ilog2_of_0_panic() {
    let _ = 0u32.ilog2();
}

#[test]
#[should_panic(expected = "argument of integer logarithm must be positive")]
fn ilog10_of_0_panic() {
    let _ = 0u32.ilog10();
}

#[test]
#[should_panic(expected = "argument of integer logarithm must be positive")]
fn ilog3_of_0_panic() {
    let _ = 0u32.ilog(3);
}

#[test]
#[should_panic(expected = "base of integer logarithm must be at least 2")]
fn ilog0_of_1_panic() {
    let _ = 1u32.ilog(0);
}

#[test]
#[should_panic(expected = "base of integer logarithm must be at least 2")]
fn ilog1_of_1_panic() {
    let _ = 1u32.ilog(1);
}
