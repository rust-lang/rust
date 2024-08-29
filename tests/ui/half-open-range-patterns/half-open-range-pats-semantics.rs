//@ run-pass

// Test half-open range patterns against their expression equivalents
// via `.contains(...)` and make sure the dynamic semantics match.

#![allow(unreachable_patterns)]
#![feature(f128)]
#![feature(f16)]

macro_rules! yes {
    ($scrutinee:expr, $($t:tt)+) => {
        {
            let m = match $scrutinee { $($t)+ => true, _ => false, };
            let c = ($($t)+).contains(&$scrutinee);
            assert_eq!(m, c);
            m
        }
    }
}

fn range_to_inclusive() {
    // `..=X` (`RangeToInclusive`-equivalent):
    //---------------------------------------

    // u8; `..=X`
    assert!(yes!(u8::MIN, ..=u8::MIN));
    assert!(yes!(u8::MIN, ..=5));
    assert!(yes!(5u8, ..=5));
    assert!(!yes!(6u8, ..=5));

    // i16; `..=X`
    assert!(yes!(i16::MIN, ..=i16::MIN));
    assert!(yes!(i16::MIN, ..=0));
    assert!(yes!(i16::MIN, ..=-5));
    assert!(yes!(-5, ..=-5));
    assert!(!yes!(-4, ..=-5));

    // char; `..=X`
    assert!(yes!('\u{0}', ..='\u{0}'));
    assert!(yes!('\u{0}', ..='a'));
    assert!(yes!('a', ..='a'));
    assert!(!yes!('b', ..='a'));

    // f16; `..=X`
    // FIXME(f16_f128): remove gate when ABI issues are resolved
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    {
        assert!(yes!(f16::NEG_INFINITY, ..=f16::NEG_INFINITY));
        assert!(yes!(f16::NEG_INFINITY, ..=1.0f16));
        assert!(yes!(1.5f16, ..=1.5f16));
        assert!(!yes!(1.6f16, ..=-1.5f16));
    }

    // f32; `..=X`
    assert!(yes!(f32::NEG_INFINITY, ..=f32::NEG_INFINITY));
    assert!(yes!(f32::NEG_INFINITY, ..=1.0f32));
    assert!(yes!(1.5f32, ..=1.5f32));
    assert!(!yes!(1.6f32, ..=-1.5f32));

    // f64; `..=X`
    assert!(yes!(f64::NEG_INFINITY, ..=f64::NEG_INFINITY));
    assert!(yes!(f64::NEG_INFINITY, ..=1.0f64));
    assert!(yes!(1.5f64, ..=1.5f64));
    assert!(!yes!(1.6f64, ..=-1.5f64));

    // f128; `..=X`
    // FIXME(f16_f128): remove gate when ABI issues are resolved
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    {
        assert!(yes!(f128::NEG_INFINITY, ..=f128::NEG_INFINITY));
        assert!(yes!(f128::NEG_INFINITY, ..=1.0f128));
        assert!(yes!(1.5f128, ..=1.5f128));
        assert!(!yes!(1.6f128, ..=-1.5f128));
    }
}

fn range_to() {
    // `..X` (`RangeTo`-equivalent):
    //-----------------------------

    // u8; `..X`
    assert!(yes!(0u8, ..1));
    assert!(yes!(0u8, ..5));
    assert!(!yes!(5u8, ..5));
    assert!(!yes!(6u8, ..5));

    // u8; `..X`
    const NU8: u8 = u8::MIN + 1;
    assert!(yes!(u8::MIN, ..NU8));
    assert!(yes!(0u8, ..5));
    assert!(!yes!(5u8, ..5));
    assert!(!yes!(6u8, ..5));

    // i16; `..X`
    const NI16: i16 = i16::MIN + 1;
    assert!(yes!(i16::MIN, ..NI16));
    assert!(yes!(i16::MIN, ..5));
    assert!(yes!(-6, ..-5));
    assert!(!yes!(-5, ..-5));

    // char; `..X`
    assert!(yes!('\u{0}', ..'\u{1}'));
    assert!(yes!('\u{0}', ..'a'));
    assert!(yes!('a', ..'b'));
    assert!(!yes!('a', ..'a'));
    assert!(!yes!('b', ..'a'));

    // f16; `..X`
    // FIXME(f16_f128): remove gate when ABI issues are resolved
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    {
        assert!(yes!(f16::NEG_INFINITY, ..1.0f16));
        assert!(!yes!(1.5f16, ..1.5f16));
        const E16: f16 = 1.5f16 + f16::EPSILON;
        assert!(yes!(1.5f16, ..E16));
        assert!(!yes!(1.6f16, ..1.5f16));
    }

    // f32; `..X`
    assert!(yes!(f32::NEG_INFINITY, ..1.0f32));
    assert!(!yes!(1.5f32, ..1.5f32));
    const E32: f32 = 1.5f32 + f32::EPSILON;
    assert!(yes!(1.5f32, ..E32));
    assert!(!yes!(1.6f32, ..1.5f32));

    // f64; `..X`
    assert!(yes!(f64::NEG_INFINITY, ..1.0f64));
    assert!(!yes!(1.5f64, ..1.5f64));
    const E64: f64 = 1.5f64 + f64::EPSILON;
    assert!(yes!(1.5f64, ..E64));
    assert!(!yes!(1.6f64, ..1.5f64));

    // f128; `..X`
    // FIXME(f16_f128): remove gate when ABI issues are resolved
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    {
        assert!(yes!(f128::NEG_INFINITY, ..1.0f128));
        assert!(!yes!(1.5f128, ..1.5f128));
        const E128: f128 = 1.5f128 + f128::EPSILON;
        assert!(yes!(1.5f128, ..E128));
        assert!(!yes!(1.6f128, ..1.5f128));
    }
}

fn range_from() {
    // `X..` (`RangeFrom`-equivalent):
    //--------------------------------

    // u8; `X..`
    assert!(yes!(u8::MIN, u8::MIN..));
    assert!(yes!(u8::MAX, u8::MIN..));
    assert!(!yes!(u8::MIN, 1..));
    assert!(!yes!(4, 5..));
    assert!(yes!(5, 5..));
    assert!(yes!(6, 5..));
    assert!(yes!(u8::MAX, u8::MAX..));

    // i16; `X..`
    assert!(yes!(i16::MIN, i16::MIN..));
    assert!(yes!(i16::MAX, i16::MIN..));
    const NI16: i16 = i16::MIN + 1;
    assert!(!yes!(i16::MIN, NI16..));
    assert!(!yes!(-4, 5..));
    assert!(yes!(-4, -4..));
    assert!(yes!(-3, -4..));
    assert!(yes!(i16::MAX, i16::MAX..));

    // char; `X..`
    assert!(yes!('\u{0}', '\u{0}'..));
    assert!(yes!(core::char::MAX, '\u{0}'..));
    assert!(yes!('a', 'a'..));
    assert!(yes!('b', 'a'..));
    assert!(!yes!('a', 'b'..));
    assert!(yes!(core::char::MAX, core::char::MAX..));

    // f16; `X..`
    // FIXME(f16_f128): remove gate when ABI issues are resolved
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    {
        assert!(yes!(f16::NEG_INFINITY, f16::NEG_INFINITY..));
        assert!(yes!(f16::INFINITY, f16::NEG_INFINITY..));
        assert!(!yes!(f16::NEG_INFINITY, 1.0f16..));
        assert!(yes!(f16::INFINITY, 1.0f16..));
        assert!(!yes!(1.0f16 - f16::EPSILON, 1.0f16..));
        assert!(yes!(1.0f16, 1.0f16..));
        assert!(yes!(f16::INFINITY, 1.0f16..));
        assert!(yes!(f16::INFINITY, f16::INFINITY..));
    }

    // f32; `X..`
    assert!(yes!(f32::NEG_INFINITY, f32::NEG_INFINITY..));
    assert!(yes!(f32::INFINITY, f32::NEG_INFINITY..));
    assert!(!yes!(f32::NEG_INFINITY, 1.0f32..));
    assert!(yes!(f32::INFINITY, 1.0f32..));
    assert!(!yes!(1.0f32 - f32::EPSILON, 1.0f32..));
    assert!(yes!(1.0f32, 1.0f32..));
    assert!(yes!(f32::INFINITY, 1.0f32..));
    assert!(yes!(f32::INFINITY, f32::INFINITY..));

    // f64; `X..`
    assert!(yes!(f64::NEG_INFINITY, f64::NEG_INFINITY..));
    assert!(yes!(f64::INFINITY, f64::NEG_INFINITY..));
    assert!(!yes!(f64::NEG_INFINITY, 1.0f64..));
    assert!(yes!(f64::INFINITY, 1.0f64..));
    assert!(!yes!(1.0f64 - f64::EPSILON, 1.0f64..));
    assert!(yes!(1.0f64, 1.0f64..));
    assert!(yes!(f64::INFINITY, 1.0f64..));
    assert!(yes!(f64::INFINITY, f64::INFINITY..));

    // f128; `X..`
    // FIXME(f16_f128): remove gate when ABI issues are resolved
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    {
        assert!(yes!(f128::NEG_INFINITY, f128::NEG_INFINITY..));
        assert!(yes!(f128::INFINITY, f128::NEG_INFINITY..));
        assert!(!yes!(f128::NEG_INFINITY, 1.0f128..));
        assert!(yes!(f128::INFINITY, 1.0f128..));
        assert!(!yes!(1.0f128 - f128::EPSILON, 1.0f128..));
        assert!(yes!(1.0f128, 1.0f128..));
        assert!(yes!(f128::INFINITY, 1.0f128..));
        assert!(yes!(f128::INFINITY, f128::INFINITY..));
    }
}

fn main() {
    range_to_inclusive();
    range_to();
    range_from();
}
