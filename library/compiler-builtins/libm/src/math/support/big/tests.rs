extern crate std;
use std::string::String;
use std::vec::Vec;
use std::{eprintln, format};

use super::{HInt, MinInt, i256, u256};

const LOHI_SPLIT: u128 = 0xaaaaaaaaaaaaaaaaffffffffffffffff;

/// Print a `u256` as hex since we can't add format implementations
fn hexu(v: u256) -> String {
    format!("0x{:016x}{:016x}{:016x}{:016x}", v.0[3], v.0[2], v.0[1], v.0[0])
}

#[test]
fn widen_u128() {
    assert_eq!(u128::MAX.widen(), u256([u64::MAX, u64::MAX, 0, 0]));
    assert_eq!(LOHI_SPLIT.widen(), u256([u64::MAX, 0xaaaaaaaaaaaaaaaa, 0, 0]));
}

#[test]
fn widen_i128() {
    assert_eq!((-1i128).widen(), u256::MAX.signed());
    assert_eq!(
        (LOHI_SPLIT as i128).widen(),
        i256([u64::MAX, 0xaaaaaaaaaaaaaaaa, u64::MAX, u64::MAX])
    );
    assert_eq!((-1i128).zero_widen().unsigned(), (u128::MAX).widen());
}

#[test]
fn widen_mul_u128() {
    let tests = [
        (u128::MAX / 2, 2_u128, u256([u64::MAX - 1, u64::MAX, 0, 0])),
        (u128::MAX, 2_u128, u256([u64::MAX - 1, u64::MAX, 1, 0])),
        (u128::MAX, u128::MAX, u256([1, 0, u64::MAX - 1, u64::MAX])),
        (u128::MIN, u128::MIN, u256::ZERO),
        (1234, 0, u256::ZERO),
        (0, 1234, u256::ZERO),
    ];

    let mut errors = Vec::new();
    for (i, (a, b, exp)) in tests.iter().copied().enumerate() {
        let res = a.widen_mul(b);
        let res_z = a.zero_widen_mul(b);
        assert_eq!(res, res_z);
        if res != exp {
            errors.push((i, a, b, exp, res));
        }
    }

    for (i, a, b, exp, res) in &errors {
        eprintln!("FAILURE ({i}): {a:#034x} * {b:#034x} = {} got {}", hexu(*exp), hexu(*res));
    }
    assert!(errors.is_empty());
}

#[test]
fn not_u128() {
    assert_eq!(!u256::ZERO, u256::MAX);
}

#[test]
fn shr_u128() {
    let only_low = [1, u16::MAX.into(), u32::MAX.into(), u64::MAX.into(), u128::MAX];

    let mut errors = Vec::new();

    for a in only_low {
        for perturb in 0..10 {
            let a = a.saturating_add(perturb);
            for shift in 0..128 {
                let res = a.widen() >> shift;
                let expected = (a >> shift).widen();
                if res != expected {
                    errors.push((a.widen(), shift, res, expected));
                }
            }
        }
    }

    let check = [
        (u256::MAX, 1, u256([u64::MAX, u64::MAX, u64::MAX, u64::MAX >> 1])),
        (u256::MAX, 5, u256([u64::MAX, u64::MAX, u64::MAX, u64::MAX >> 5])),
        (u256::MAX, 63, u256([u64::MAX, u64::MAX, u64::MAX, 1])),
        (u256::MAX, 64, u256([u64::MAX, u64::MAX, u64::MAX, 0])),
        (u256::MAX, 65, u256([u64::MAX, u64::MAX, u64::MAX >> 1, 0])),
        (u256::MAX, 127, u256([u64::MAX, u64::MAX, 1, 0])),
        (u256::MAX, 128, u256([u64::MAX, u64::MAX, 0, 0])),
        (u256::MAX, 129, u256([u64::MAX, u64::MAX >> 1, 0, 0])),
        (u256::MAX, 191, u256([u64::MAX, 1, 0, 0])),
        (u256::MAX, 192, u256([u64::MAX, 0, 0, 0])),
        (u256::MAX, 193, u256([u64::MAX >> 1, 0, 0, 0])),
        (u256::MAX, 191, u256([u64::MAX, 1, 0, 0])),
        (u256::MAX, 254, u256([0b11, 0, 0, 0])),
        (u256::MAX, 255, u256([1, 0, 0, 0])),
    ];

    for (input, shift, expected) in check {
        let res = input >> shift;
        if res != expected {
            errors.push((input, shift, res, expected));
        }
    }

    for (a, b, res, expected) in &errors {
        eprintln!("FAILURE: {} >> {b} = {} got {}", hexu(*a), hexu(*expected), hexu(*res),);
    }
    assert!(errors.is_empty());
}

#[test]
#[should_panic]
#[cfg(debug_assertions)]
// FIXME(ppc): ppc64le seems to have issues with `should_panic` tests.
#[cfg(not(all(target_arch = "powerpc64", target_endian = "little")))]
fn shr_u256_overflow() {
    // Like regular shr, panic on overflow with debug assertions
    let _ = u256::MAX >> 256;
}

#[test]
#[cfg(not(debug_assertions))]
fn shr_u256_overflow() {
    // No panic without debug assertions
    assert_eq!(u256::MAX >> 256, u256::ZERO);
    assert_eq!(u256::MAX >> 257, u256::ZERO);
    assert_eq!(u256::MAX >> u32::MAX, u256::ZERO);
}
