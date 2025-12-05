use core::num::bignum::Big32x40;
use core::num::bignum::tests::Big8x3 as Big;

#[test]
#[should_panic]
fn test_from_u64_overflow() {
    Big::from_u64(0x1000000);
}

#[test]
fn test_add() {
    assert_eq!(*Big::from_small(3).add(&Big::from_small(4)), Big::from_small(7));
    assert_eq!(*Big::from_small(3).add(&Big::from_small(0)), Big::from_small(3));
    assert_eq!(*Big::from_small(0).add(&Big::from_small(3)), Big::from_small(3));
    assert_eq!(*Big::from_small(3).add(&Big::from_u64(0xfffe)), Big::from_u64(0x10001));
    assert_eq!(*Big::from_u64(0xfedc).add(&Big::from_u64(0x789)), Big::from_u64(0x10665));
    assert_eq!(*Big::from_u64(0x789).add(&Big::from_u64(0xfedc)), Big::from_u64(0x10665));
}

#[test]
#[should_panic]
fn test_add_overflow_1() {
    Big::from_small(1).add(&Big::from_u64(0xffffff));
}

#[test]
#[should_panic]
fn test_add_overflow_2() {
    Big::from_u64(0xffffff).add(&Big::from_small(1));
}

#[test]
fn test_add_small() {
    assert_eq!(*Big::from_small(3).add_small(4), Big::from_small(7));
    assert_eq!(*Big::from_small(3).add_small(0), Big::from_small(3));
    assert_eq!(*Big::from_small(0).add_small(3), Big::from_small(3));
    assert_eq!(*Big::from_small(7).add_small(250), Big::from_u64(257));
    assert_eq!(*Big::from_u64(0x7fff).add_small(1), Big::from_u64(0x8000));
    assert_eq!(*Big::from_u64(0x2ffe).add_small(0x35), Big::from_u64(0x3033));
    assert_eq!(*Big::from_small(0xdc).add_small(0x89), Big::from_u64(0x165));
}

#[test]
#[should_panic]
fn test_add_small_overflow() {
    Big::from_u64(0xffffff).add_small(1);
}

#[test]
fn test_sub() {
    assert_eq!(*Big::from_small(7).sub(&Big::from_small(4)), Big::from_small(3));
    assert_eq!(*Big::from_u64(0x10665).sub(&Big::from_u64(0x789)), Big::from_u64(0xfedc));
    assert_eq!(*Big::from_u64(0x10665).sub(&Big::from_u64(0xfedc)), Big::from_u64(0x789));
    assert_eq!(*Big::from_u64(0x10665).sub(&Big::from_u64(0x10664)), Big::from_small(1));
    assert_eq!(*Big::from_u64(0x10665).sub(&Big::from_u64(0x10665)), Big::from_small(0));
}

#[test]
#[should_panic]
fn test_sub_underflow_1() {
    Big::from_u64(0x10665).sub(&Big::from_u64(0x10666));
}

#[test]
#[should_panic]
fn test_sub_underflow_2() {
    Big::from_small(0).sub(&Big::from_u64(0x123456));
}

#[test]
fn test_mul_small() {
    assert_eq!(*Big::from_small(7).mul_small(5), Big::from_small(35));
    assert_eq!(*Big::from_small(0xff).mul_small(0xff), Big::from_u64(0xfe01));
    assert_eq!(*Big::from_u64(0xffffff / 13).mul_small(13), Big::from_u64(0xffffff));
}

#[test]
#[should_panic]
fn test_mul_small_overflow() {
    Big::from_u64(0x800000).mul_small(2);
}

#[test]
fn test_mul_pow2() {
    assert_eq!(*Big::from_small(0x7).mul_pow2(4), Big::from_small(0x70));
    assert_eq!(*Big::from_small(0xff).mul_pow2(1), Big::from_u64(0x1fe));
    assert_eq!(*Big::from_small(0xff).mul_pow2(12), Big::from_u64(0xff000));
    assert_eq!(*Big::from_small(0x1).mul_pow2(23), Big::from_u64(0x800000));
    assert_eq!(*Big::from_u64(0x123).mul_pow2(0), Big::from_u64(0x123));
    assert_eq!(*Big::from_u64(0x123).mul_pow2(7), Big::from_u64(0x9180));
    assert_eq!(*Big::from_u64(0x123).mul_pow2(15), Big::from_u64(0x918000));
    assert_eq!(*Big::from_small(0).mul_pow2(23), Big::from_small(0));
}

#[test]
#[should_panic]
fn test_mul_pow2_overflow_1() {
    Big::from_u64(0x1).mul_pow2(24);
}

#[test]
#[should_panic]
fn test_mul_pow2_overflow_2() {
    Big::from_u64(0x123).mul_pow2(16);
}

#[test]
fn test_mul_pow5() {
    assert_eq!(*Big::from_small(42).mul_pow5(0), Big::from_small(42));
    assert_eq!(*Big::from_small(1).mul_pow5(2), Big::from_small(25));
    assert_eq!(*Big::from_small(1).mul_pow5(4), Big::from_u64(25 * 25));
    assert_eq!(*Big::from_small(4).mul_pow5(3), Big::from_u64(500));
    assert_eq!(*Big::from_small(140).mul_pow5(2), Big::from_u64(25 * 140));
    assert_eq!(*Big::from_small(25).mul_pow5(1), Big::from_small(125));
    assert_eq!(*Big::from_small(125).mul_pow5(7), Big::from_u64(9765625));
    assert_eq!(*Big::from_small(0).mul_pow5(127), Big::from_small(0));
}

#[test]
#[should_panic]
fn test_mul_pow5_overflow_1() {
    Big::from_small(1).mul_pow5(12);
}

#[test]
#[should_panic]
fn test_mul_pow5_overflow_2() {
    Big::from_small(230).mul_pow5(8);
}

#[test]
fn test_mul_digits() {
    assert_eq!(*Big::from_small(3).mul_digits(&[5]), Big::from_small(15));
    assert_eq!(*Big::from_small(0xff).mul_digits(&[0xff]), Big::from_u64(0xfe01));
    assert_eq!(*Big::from_u64(0x123).mul_digits(&[0x56, 0x4]), Big::from_u64(0x4edc2));
    assert_eq!(*Big::from_u64(0x12345).mul_digits(&[0x67]), Big::from_u64(0x7530c3));
    assert_eq!(*Big::from_small(0x12).mul_digits(&[0x67, 0x45, 0x3]), Big::from_u64(0x3ae13e));
    assert_eq!(*Big::from_u64(0xffffff / 13).mul_digits(&[13]), Big::from_u64(0xffffff));
    assert_eq!(*Big::from_small(13).mul_digits(&[0x3b, 0xb1, 0x13]), Big::from_u64(0xffffff));
}

#[test]
#[should_panic]
fn test_mul_digits_overflow_1() {
    Big::from_u64(0x800000).mul_digits(&[2]);
}

#[test]
#[should_panic]
fn test_mul_digits_overflow_2() {
    Big::from_u64(0x1000).mul_digits(&[0, 0x10]);
}

#[test]
fn test_div_rem_small() {
    let as_val = |(q, r): (&mut Big, u8)| (q.clone(), r);
    assert_eq!(as_val(Big::from_small(0xff).div_rem_small(15)), (Big::from_small(17), 0));
    assert_eq!(as_val(Big::from_small(0xff).div_rem_small(16)), (Big::from_small(15), 15));
    assert_eq!(as_val(Big::from_small(3).div_rem_small(40)), (Big::from_small(0), 3));
    assert_eq!(
        as_val(Big::from_u64(0xffffff).div_rem_small(123)),
        (Big::from_u64(0xffffff / 123), (0xffffffu64 % 123) as u8)
    );
    assert_eq!(
        as_val(Big::from_u64(0x10000).div_rem_small(123)),
        (Big::from_u64(0x10000 / 123), (0x10000u64 % 123) as u8)
    );
}

#[test]
fn test_is_zero() {
    assert!(Big::from_small(0).is_zero());
    assert!(!Big::from_small(3).is_zero());
    assert!(!Big::from_u64(0x123).is_zero());
    assert!(!Big::from_u64(0xffffff).sub(&Big::from_u64(0xfffffe)).is_zero());
    assert!(Big::from_u64(0xffffff).sub(&Big::from_u64(0xffffff)).is_zero());
}

#[test]
fn test_get_bit() {
    let x = Big::from_small(0b1101);
    assert_eq!(x.get_bit(0), 1);
    assert_eq!(x.get_bit(1), 0);
    assert_eq!(x.get_bit(2), 1);
    assert_eq!(x.get_bit(3), 1);
    let y = Big::from_u64(1 << 15);
    assert_eq!(y.get_bit(14), 0);
    assert_eq!(y.get_bit(15), 1);
    assert_eq!(y.get_bit(16), 0);
}

#[test]
#[should_panic]
fn test_get_bit_out_of_range() {
    Big::from_small(42).get_bit(24);
}

#[test]
fn test_bit_length() {
    for i in 0..8 * 3 {
        // 010000...000
        assert_eq!(Big::from_small(1).mul_pow2(i).bit_length(), i + 1);
    }
    for i in 1..8 * 3 - 1 {
        // 010000...001
        assert_eq!(Big::from_small(1).mul_pow2(i).add(&Big::from_small(1)).bit_length(), i + 1);
        // 110000...000
        assert_eq!(Big::from_small(3).mul_pow2(i).bit_length(), i + 2);
    }
    assert_eq!(Big::from_small(0).bit_length(), 0);
    assert_eq!(Big::from_small(1).bit_length(), 1);
    assert_eq!(Big::from_small(5).bit_length(), 3);
    assert_eq!(Big::from_small(0x18).bit_length(), 5);
    assert_eq!(Big::from_u64(0x4073).bit_length(), 15);
    assert_eq!(Big::from_u64(0xffffff).bit_length(), 24);
}

#[test]
fn test_bit_length_32x40() {
    for i in 0..32 * 40 {
        // 010000...000
        assert_eq!(Big32x40::from_small(1).mul_pow2(i).bit_length(), i + 1);
    }
    for i in 1..32 * 40 - 1 {
        // 010000...001
        assert_eq!(
            Big32x40::from_small(1).mul_pow2(i).add(&Big32x40::from_small(1)).bit_length(),
            i + 1
        );
        // 110000...000
        assert_eq!(Big32x40::from_small(3).mul_pow2(i).bit_length(), i + 2);
    }
    assert_eq!(Big32x40::from_small(0).bit_length(), 0);
    assert_eq!(Big32x40::from_small(1).bit_length(), 1);
    assert_eq!(Big32x40::from_small(5).bit_length(), 3);
    assert_eq!(Big32x40::from_small(0x18).bit_length(), 5);
    assert_eq!(Big32x40::from_u64(0x4073).bit_length(), 15);
    assert_eq!(Big32x40::from_u64(0xffffff).bit_length(), 24);
    assert_eq!(Big32x40::from_u64(0xffffffffffffffff).bit_length(), 64);
}

#[test]
fn test_ord() {
    assert!(Big::from_u64(0) < Big::from_u64(0xffffff));
    assert!(Big::from_u64(0x102) < Big::from_u64(0x201));
}

#[test]
fn test_fmt() {
    assert_eq!(format!("{:?}", Big::from_u64(0)), "0x0");
    assert_eq!(format!("{:?}", Big::from_u64(0x1)), "0x1");
    assert_eq!(format!("{:?}", Big::from_u64(0x12)), "0x12");
    assert_eq!(format!("{:?}", Big::from_u64(0x123)), "0x1_23");
    assert_eq!(format!("{:?}", Big::from_u64(0x1234)), "0x12_34");
    assert_eq!(format!("{:?}", Big::from_u64(0x12345)), "0x1_23_45");
    assert_eq!(format!("{:?}", Big::from_u64(0x123456)), "0x12_34_56");
}
