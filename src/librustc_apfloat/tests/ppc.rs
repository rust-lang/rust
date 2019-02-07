use rustc_apfloat::{Category, Float, Round};
use rustc_apfloat::ppc::DoubleDouble;

use std::cmp::Ordering;

#[test]
fn ppc_double_double() {
    let test = DoubleDouble::ZERO;
    let expected = "0x0p+0".parse::<DoubleDouble>().unwrap();
    assert!(test.is_zero());
    assert!(!test.is_negative());
    assert!(test.bitwise_eq(expected));
    assert_eq!(0, test.to_bits());

    let test = -DoubleDouble::ZERO;
    let expected = "-0x0p+0".parse::<DoubleDouble>().unwrap();
    assert!(test.is_zero());
    assert!(test.is_negative());
    assert!(test.bitwise_eq(expected));
    assert_eq!(0x8000000000000000, test.to_bits());

    let test = "1.0".parse::<DoubleDouble>().unwrap();
    assert_eq!(0x3ff0000000000000, test.to_bits());

    // LDBL_MAX
    let test = "1.79769313486231580793728971405301e+308"
        .parse::<DoubleDouble>()
        .unwrap();
    assert_eq!(0x7c8ffffffffffffe_7fefffffffffffff, test.to_bits());

    // LDBL_MIN
    let test = "2.00416836000897277799610805135016e-292"
        .parse::<DoubleDouble>()
        .unwrap();
    assert_eq!(0x0000000000000000_0360000000000000, test.to_bits());
}

#[test]
fn ppc_double_double_add_special() {
    let data = [
        // (1 + 0) + (-1 + 0) = Category::Zero
        (
            0x3ff0000000000000,
            0xbff0000000000000,
            Category::Zero,
            Round::NearestTiesToEven,
        ),
        // LDBL_MAX + (1.1 >> (1023 - 106) + 0)) = Category::Infinity
        (
            0x7c8ffffffffffffe_7fefffffffffffff,
            0x7948000000000000,
            Category::Infinity,
            Round::NearestTiesToEven,
        ),
        // FIXME: change the 4th 0x75effffffffffffe to 0x75efffffffffffff when
        // DoubleDouble's fallback is gone.
        // LDBL_MAX + (1.011111... >> (1023 - 106) + (1.1111111...0 >> (1023 -
        // 160))) = Category::Normal
        (
            0x7c8ffffffffffffe_7fefffffffffffff,
            0x75effffffffffffe_7947ffffffffffff,
            Category::Normal,
            Round::NearestTiesToEven,
        ),
        // LDBL_MAX + (1.1 >> (1023 - 106) + 0)) = Category::Infinity
        (
            0x7c8ffffffffffffe_7fefffffffffffff,
            0x7c8ffffffffffffe_7fefffffffffffff,
            Category::Infinity,
            Round::NearestTiesToEven,
        ),
        // NaN + (1 + 0) = Category::NaN
        (
            0x7ff8000000000000,
            0x3ff0000000000000,
            Category::NaN,
            Round::NearestTiesToEven,
        ),
    ];

    for &(op1, op2, expected, round) in &data {
        {
            let mut a1 = DoubleDouble::from_bits(op1);
            let a2 = DoubleDouble::from_bits(op2);
            a1 = a1.add_r(a2, round).value;

            assert_eq!(expected, a1.category(), "{:#x} + {:#x}", op1, op2);
        }
        {
            let a1 = DoubleDouble::from_bits(op1);
            let mut a2 = DoubleDouble::from_bits(op2);
            a2 = a2.add_r(a1, round).value;

            assert_eq!(expected, a2.category(), "{:#x} + {:#x}", op2, op1);
        }
    }
}

#[test]
fn ppc_double_double_add() {
    let data = [
        // (1 + 0) + (1e-105 + 0) = (1 + 1e-105)
        (
            0x3ff0000000000000,
            0x3960000000000000,
            0x3960000000000000_3ff0000000000000,
            Round::NearestTiesToEven,
        ),
        // (1 + 0) + (1e-106 + 0) = (1 + 1e-106)
        (
            0x3ff0000000000000,
            0x3950000000000000,
            0x3950000000000000_3ff0000000000000,
            Round::NearestTiesToEven,
        ),
        // (1 + 1e-106) + (1e-106 + 0) = (1 + 1e-105)
        (
            0x3950000000000000_3ff0000000000000,
            0x3950000000000000,
            0x3960000000000000_3ff0000000000000,
            Round::NearestTiesToEven,
        ),
        // (1 + 0) + (epsilon + 0) = (1 + epsilon)
        (
            0x3ff0000000000000,
            0x0000000000000001,
            0x0000000000000001_3ff0000000000000,
            Round::NearestTiesToEven,
        ),
        // FIXME: change 0xf950000000000000 to 0xf940000000000000, when
        // DoubleDouble's fallback is gone.
        // (DBL_MAX - 1 << (1023 - 105)) + (1 << (1023 - 53) + 0) = DBL_MAX +
        // 1.11111... << (1023 - 52)
        (
            0xf950000000000000_7fefffffffffffff,
            0x7c90000000000000,
            0x7c8ffffffffffffe_7fefffffffffffff,
            Round::NearestTiesToEven,
        ),
        // FIXME: change 0xf950000000000000 to 0xf940000000000000, when
        // DoubleDouble's fallback is gone.
        // (1 << (1023 - 53) + 0) + (DBL_MAX - 1 << (1023 - 105)) = DBL_MAX +
        // 1.11111... << (1023 - 52)
        (
            0x7c90000000000000,
            0xf950000000000000_7fefffffffffffff,
            0x7c8ffffffffffffe_7fefffffffffffff,
            Round::NearestTiesToEven,
        ),
    ];

    for &(op1, op2, expected, round) in &data {
        {
            let mut a1 = DoubleDouble::from_bits(op1);
            let a2 = DoubleDouble::from_bits(op2);
            a1 = a1.add_r(a2, round).value;

            assert_eq!(expected, a1.to_bits(), "{:#x} + {:#x}", op1, op2);
        }
        {
            let a1 = DoubleDouble::from_bits(op1);
            let mut a2 = DoubleDouble::from_bits(op2);
            a2 = a2.add_r(a1, round).value;

            assert_eq!(expected, a2.to_bits(), "{:#x} + {:#x}", op2, op1);
        }
    }
}

#[test]
fn ppc_double_double_subtract() {
    let data = [
        // (1 + 0) - (-1e-105 + 0) = (1 + 1e-105)
        (
            0x3ff0000000000000,
            0xb960000000000000,
            0x3960000000000000_3ff0000000000000,
            Round::NearestTiesToEven,
        ),
        // (1 + 0) - (-1e-106 + 0) = (1 + 1e-106)
        (
            0x3ff0000000000000,
            0xb950000000000000,
            0x3950000000000000_3ff0000000000000,
            Round::NearestTiesToEven,
        ),
    ];

    for &(op1, op2, expected, round) in &data {
        let mut a1 = DoubleDouble::from_bits(op1);
        let a2 = DoubleDouble::from_bits(op2);
        a1 = a1.sub_r(a2, round).value;

        assert_eq!(expected, a1.to_bits(), "{:#x} - {:#x}", op1, op2);
    }
}

#[test]
fn ppc_double_double_multiply_special() {
    let data = [
        // Category::NaN * Category::NaN = Category::NaN
        (
            0x7ff8000000000000,
            0x7ff8000000000000,
            Category::NaN,
            Round::NearestTiesToEven,
        ),
        // Category::NaN * Category::Zero = Category::NaN
        (
            0x7ff8000000000000,
            0,
            Category::NaN,
            Round::NearestTiesToEven,
        ),
        // Category::NaN * Category::Infinity = Category::NaN
        (
            0x7ff8000000000000,
            0x7ff0000000000000,
            Category::NaN,
            Round::NearestTiesToEven,
        ),
        // Category::NaN * Category::Normal = Category::NaN
        (
            0x7ff8000000000000,
            0x3ff0000000000000,
            Category::NaN,
            Round::NearestTiesToEven,
        ),
        // Category::Infinity * Category::Infinity = Category::Infinity
        (
            0x7ff0000000000000,
            0x7ff0000000000000,
            Category::Infinity,
            Round::NearestTiesToEven,
        ),
        // Category::Infinity * Category::Zero = Category::NaN
        (
            0x7ff0000000000000,
            0,
            Category::NaN,
            Round::NearestTiesToEven,
        ),
        // Category::Infinity * Category::Normal = Category::Infinity
        (
            0x7ff0000000000000,
            0x3ff0000000000000,
            Category::Infinity,
            Round::NearestTiesToEven,
        ),
        // Category::Zero * Category::Zero = Category::Zero
        (0, 0, Category::Zero, Round::NearestTiesToEven),
        // Category::Zero * Category::Normal = Category::Zero
        (
            0,
            0x3ff0000000000000,
            Category::Zero,
            Round::NearestTiesToEven,
        ),
    ];

    for &(op1, op2, expected, round) in &data {
        {
            let mut a1 = DoubleDouble::from_bits(op1);
            let a2 = DoubleDouble::from_bits(op2);
            a1 = a1.mul_r(a2, round).value;

            assert_eq!(expected, a1.category(), "{:#x} * {:#x}", op1, op2);
        }
        {
            let a1 = DoubleDouble::from_bits(op1);
            let mut a2 = DoubleDouble::from_bits(op2);
            a2 = a2.mul_r(a1, round).value;

            assert_eq!(expected, a2.category(), "{:#x} * {:#x}", op2, op1);
        }
    }
}

#[test]
fn ppc_double_double_multiply() {
    let data = [
        // 1/3 * 3 = 1.0
        (
            0x3c75555555555556_3fd5555555555555,
            0x4008000000000000,
            0x3ff0000000000000,
            Round::NearestTiesToEven,
        ),
        // (1 + epsilon) * (1 + 0) = Category::Zero
        (
            0x0000000000000001_3ff0000000000000,
            0x3ff0000000000000,
            0x0000000000000001_3ff0000000000000,
            Round::NearestTiesToEven,
        ),
        // (1 + epsilon) * (1 + epsilon) = 1 + 2 * epsilon
        (
            0x0000000000000001_3ff0000000000000,
            0x0000000000000001_3ff0000000000000,
            0x0000000000000002_3ff0000000000000,
            Round::NearestTiesToEven,
        ),
        // -(1 + epsilon) * (1 + epsilon) = -1
        (
            0x0000000000000001_bff0000000000000,
            0x0000000000000001_3ff0000000000000,
            0xbff0000000000000,
            Round::NearestTiesToEven,
        ),
        // (0.5 + 0) * (1 + 2 * epsilon) = 0.5 + epsilon
        (
            0x3fe0000000000000,
            0x0000000000000002_3ff0000000000000,
            0x0000000000000001_3fe0000000000000,
            Round::NearestTiesToEven,
        ),
        // (0.5 + 0) * (1 + epsilon) = 0.5
        (
            0x3fe0000000000000,
            0x0000000000000001_3ff0000000000000,
            0x3fe0000000000000,
            Round::NearestTiesToEven,
        ),
        // __LDBL_MAX__ * (1 + 1 << 106) = inf
        (
            0x7c8ffffffffffffe_7fefffffffffffff,
            0x3950000000000000_3ff0000000000000,
            0x7ff0000000000000,
            Round::NearestTiesToEven,
        ),
        // __LDBL_MAX__ * (1 + 1 << 107) > __LDBL_MAX__, but not inf, yes =_=|||
        (
            0x7c8ffffffffffffe_7fefffffffffffff,
            0x3940000000000000_3ff0000000000000,
            0x7c8fffffffffffff_7fefffffffffffff,
            Round::NearestTiesToEven,
        ),
        // __LDBL_MAX__ * (1 + 1 << 108) = __LDBL_MAX__
        (
            0x7c8ffffffffffffe_7fefffffffffffff,
            0x3930000000000000_3ff0000000000000,
            0x7c8ffffffffffffe_7fefffffffffffff,
            Round::NearestTiesToEven,
        ),
    ];

    for &(op1, op2, expected, round) in &data {
        {
            let mut a1 = DoubleDouble::from_bits(op1);
            let a2 = DoubleDouble::from_bits(op2);
            a1 = a1.mul_r(a2, round).value;

            assert_eq!(expected, a1.to_bits(), "{:#x} * {:#x}", op1, op2);
        }
        {
            let a1 = DoubleDouble::from_bits(op1);
            let mut a2 = DoubleDouble::from_bits(op2);
            a2 = a2.mul_r(a1, round).value;

            assert_eq!(expected, a2.to_bits(), "{:#x} * {:#x}", op2, op1);
        }
    }
}

#[test]
fn ppc_double_double_divide() {
    // FIXME: Only a sanity check for now. Add more edge cases when the
    // double-double algorithm is implemented.
    let data = [
        // 1 / 3 = 1/3
        (
            0x3ff0000000000000,
            0x4008000000000000,
            0x3c75555555555556_3fd5555555555555,
            Round::NearestTiesToEven,
        ),
    ];

    for &(op1, op2, expected, round) in &data {
        let mut a1 = DoubleDouble::from_bits(op1);
        let a2 = DoubleDouble::from_bits(op2);
        a1 = a1.div_r(a2, round).value;

        assert_eq!(expected, a1.to_bits(), "{:#x} / {:#x}", op1, op2);
    }
}

#[test]
fn ppc_double_double_remainder() {
    let data = [
        // ieee_rem(3.0 + 3.0 << 53, 1.25 + 1.25 << 53) = (0.5 + 0.5 << 53)
        (
            0x3cb8000000000000_4008000000000000,
            0x3ca4000000000000_3ff4000000000000,
            0x3c90000000000000_3fe0000000000000,
        ),
        // ieee_rem(3.0 + 3.0 << 53, 1.75 + 1.75 << 53) = (-0.5 - 0.5 << 53)
        (
            0x3cb8000000000000_4008000000000000,
            0x3cac000000000000_3ffc000000000000,
            0xbc90000000000000_bfe0000000000000,
        ),
    ];

    for &(op1, op2, expected) in &data {
        let a1 = DoubleDouble::from_bits(op1);
        let a2 = DoubleDouble::from_bits(op2);
        let result = a1.ieee_rem(a2).value;

        assert_eq!(
            expected,
            result.to_bits(),
            "ieee_rem({:#x}, {:#x})",
            op1,
            op2
        );
    }
}

#[test]
fn ppc_double_double_mod() {
    let data = [
        // mod(3.0 + 3.0 << 53, 1.25 + 1.25 << 53) = (0.5 + 0.5 << 53)
        (
            0x3cb8000000000000_4008000000000000,
            0x3ca4000000000000_3ff4000000000000,
            0x3c90000000000000_3fe0000000000000,
        ),
        // mod(3.0 + 3.0 << 53, 1.75 + 1.75 << 53) = (1.25 + 1.25 << 53)
        // 0xbc98000000000000 doesn't seem right, but it's what we currently have.
        // FIXME: investigate
        (
            0x3cb8000000000000_4008000000000000,
            0x3cac000000000000_3ffc000000000000,
            0xbc98000000000000_3ff4000000000001,
        ),
    ];

    for &(op1, op2, expected) in &data {
        let a1 = DoubleDouble::from_bits(op1);
        let a2 = DoubleDouble::from_bits(op2);
        let r = (a1 % a2).value;

        assert_eq!(expected, r.to_bits(), "fmod({:#x}, {:#x})", op1, op2);
    }
}

#[test]
fn ppc_double_double_fma() {
    // Sanity check for now.
    let mut a = "2".parse::<DoubleDouble>().unwrap();
    a = a.mul_add(
        "3".parse::<DoubleDouble>().unwrap(),
        "4".parse::<DoubleDouble>().unwrap(),
    ).value;
    assert_eq!(
        Some(Ordering::Equal),
        "10".parse::<DoubleDouble>().unwrap().partial_cmp(&a)
    );
}

#[test]
fn ppc_double_double_round_to_integral() {
    {
        let a = "1.5".parse::<DoubleDouble>().unwrap();
        let a = a.round_to_integral(Round::NearestTiesToEven).value;
        assert_eq!(
            Some(Ordering::Equal),
            "2".parse::<DoubleDouble>().unwrap().partial_cmp(&a)
        );
    }
    {
        let a = "2.5".parse::<DoubleDouble>().unwrap();
        let a = a.round_to_integral(Round::NearestTiesToEven).value;
        assert_eq!(
            Some(Ordering::Equal),
            "2".parse::<DoubleDouble>().unwrap().partial_cmp(&a)
        );
    }
}

#[test]
fn ppc_double_double_compare() {
    let data = [
        // (1 + 0) = (1 + 0)
        (
            0x3ff0000000000000,
            0x3ff0000000000000,
            Some(Ordering::Equal),
        ),
        // (1 + 0) < (1.00...1 + 0)
        (0x3ff0000000000000, 0x3ff0000000000001, Some(Ordering::Less)),
        // (1.00...1 + 0) > (1 + 0)
        (
            0x3ff0000000000001,
            0x3ff0000000000000,
            Some(Ordering::Greater),
        ),
        // (1 + 0) < (1 + epsilon)
        (
            0x3ff0000000000000,
            0x0000000000000001_3ff0000000000001,
            Some(Ordering::Less),
        ),
        // NaN != NaN
        (0x7ff8000000000000, 0x7ff8000000000000, None),
        // (1 + 0) != NaN
        (0x3ff0000000000000, 0x7ff8000000000000, None),
        // Inf = Inf
        (
            0x7ff0000000000000,
            0x7ff0000000000000,
            Some(Ordering::Equal),
        ),
    ];

    for &(op1, op2, expected) in &data {
        let a1 = DoubleDouble::from_bits(op1);
        let a2 = DoubleDouble::from_bits(op2);
        assert_eq!(
            expected,
            a1.partial_cmp(&a2),
            "compare({:#x}, {:#x})",
            op1,
            op2,
        );
    }
}

#[test]
fn ppc_double_double_bitwise_eq() {
    let data = [
        // (1 + 0) = (1 + 0)
        (0x3ff0000000000000, 0x3ff0000000000000, true),
        // (1 + 0) != (1.00...1 + 0)
        (0x3ff0000000000000, 0x3ff0000000000001, false),
        // NaN = NaN
        (0x7ff8000000000000, 0x7ff8000000000000, true),
        // NaN != NaN with a different bit pattern
        (
            0x7ff8000000000000,
            0x3ff0000000000000_7ff8000000000000,
            false,
        ),
        // Inf = Inf
        (0x7ff0000000000000, 0x7ff0000000000000, true),
    ];

    for &(op1, op2, expected) in &data {
        let a1 = DoubleDouble::from_bits(op1);
        let a2 = DoubleDouble::from_bits(op2);
        assert_eq!(expected, a1.bitwise_eq(a2), "{:#x} = {:#x}", op1, op2);
    }
}

#[test]
fn ppc_double_double_change_sign() {
    let float = DoubleDouble::from_bits(0xbcb0000000000000_400f000000000000);
    {
        let actual = float.copy_sign("1".parse::<DoubleDouble>().unwrap());
        assert_eq!(0xbcb0000000000000_400f000000000000, actual.to_bits());
    }
    {
        let actual = float.copy_sign("-1".parse::<DoubleDouble>().unwrap());
        assert_eq!(0x3cb0000000000000_c00f000000000000, actual.to_bits());
    }
}

#[test]
fn ppc_double_double_factories() {
    assert_eq!(0, DoubleDouble::ZERO.to_bits());
    assert_eq!(
        0x7c8ffffffffffffe_7fefffffffffffff,
        DoubleDouble::largest().to_bits()
    );
    assert_eq!(0x0000000000000001, DoubleDouble::SMALLEST.to_bits());
    assert_eq!(
        0x0360000000000000,
        DoubleDouble::smallest_normalized().to_bits()
    );
    assert_eq!(
        0x0000000000000000_8000000000000000,
        (-DoubleDouble::ZERO).to_bits()
    );
    assert_eq!(
        0xfc8ffffffffffffe_ffefffffffffffff,
        (-DoubleDouble::largest()).to_bits()
    );
    assert_eq!(
        0x0000000000000000_8000000000000001,
        (-DoubleDouble::SMALLEST).to_bits()
    );
    assert_eq!(
        0x0000000000000000_8360000000000000,
        (-DoubleDouble::smallest_normalized()).to_bits()
    );
    assert!(DoubleDouble::SMALLEST.is_smallest());
    assert!(DoubleDouble::largest().is_largest());
}

#[test]
fn ppc_double_double_is_denormal() {
    assert!(DoubleDouble::SMALLEST.is_denormal());
    assert!(!DoubleDouble::largest().is_denormal());
    assert!(!DoubleDouble::smallest_normalized().is_denormal());
    {
        // (4 + 3) is not normalized
        let data = 0x4008000000000000_4010000000000000;
        assert!(DoubleDouble::from_bits(data).is_denormal());
    }
}

#[test]
fn ppc_double_double_exact_inverse() {
    assert!(
        "2.0"
            .parse::<DoubleDouble>()
            .unwrap()
            .get_exact_inverse()
            .unwrap()
            .bitwise_eq("0.5".parse::<DoubleDouble>().unwrap())
    );
}

#[test]
fn ppc_double_double_scalbn() {
    // 3.0 + 3.0 << 53
    let input = 0x3cb8000000000000_4008000000000000;
    let result = DoubleDouble::from_bits(input).scalbn(1);
    // 6.0 + 6.0 << 53
    assert_eq!(0x3cc8000000000000_4018000000000000, result.to_bits());
}

#[test]
fn ppc_double_double_frexp() {
    // 3.0 + 3.0 << 53
    let input = 0x3cb8000000000000_4008000000000000;
    let mut exp = 0;
    // 0.75 + 0.75 << 53
    let result = DoubleDouble::from_bits(input).frexp(&mut exp);
    assert_eq!(2, exp);
    assert_eq!(0x3c98000000000000_3fe8000000000000, result.to_bits());
}
