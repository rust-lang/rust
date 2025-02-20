use core::num::{IntErrorKind, NonZero};
use core::option::Option::None;
use std::mem::size_of;

#[test]
fn test_create_nonzero_instance() {
    let _a = unsafe { NonZero::new_unchecked(21) };
}

#[test]
fn test_size_nonzero_in_option() {
    assert_eq!(size_of::<NonZero<u32>>(), size_of::<Option<NonZero<u32>>>());
    assert_eq!(size_of::<NonZero<i32>>(), size_of::<Option<NonZero<i32>>>());
}

#[test]
fn test_match_on_nonzero_option() {
    let a = Some(unsafe { NonZero::<u32>::new_unchecked(42) });
    match a {
        Some(val) => assert_eq!(val.get(), 42),
        None => panic!("unexpected None while matching on Some(NonZero(_))"),
    }

    match unsafe { Some(NonZero::<u32>::new_unchecked(43)) } {
        Some(val) => assert_eq!(val.get(), 43),
        None => panic!("unexpected None while matching on Some(NonZero(_))"),
    }
}

#[test]
fn test_match_option_empty_vec() {
    let a: Option<Vec<isize>> = Some(vec![]);
    match a {
        None => panic!("unexpected None while matching on Some(vec![])"),
        _ => {}
    }
}

#[test]
fn test_match_option_vec() {
    let a = Some(vec![1, 2, 3, 4]);
    match a {
        Some(v) => assert_eq!(v, [1, 2, 3, 4]),
        None => panic!("unexpected None while matching on Some(vec![1, 2, 3, 4])"),
    }
}

#[test]
fn test_match_option_rc() {
    use std::rc::Rc;

    let five = Rc::new(5);
    match Some(five) {
        Some(r) => assert_eq!(*r, 5),
        None => panic!("unexpected None while matching on Some(Rc::new(5))"),
    }
}

#[test]
fn test_match_option_arc() {
    use std::sync::Arc;

    let five = Arc::new(5);
    match Some(five) {
        Some(a) => assert_eq!(*a, 5),
        None => panic!("unexpected None while matching on Some(Arc::new(5))"),
    }
}

#[test]
fn test_match_option_empty_string() {
    let a = Some(String::new());
    match a {
        None => panic!("unexpected None while matching on Some(String::new())"),
        _ => {}
    }
}

#[test]
fn test_match_option_string() {
    let five = "Five".to_string();
    match Some(five) {
        Some(s) => assert_eq!(s, "Five"),
        None => panic!("{}", "unexpected None while matching on Some(String { ... })"),
    }
}

mod atom {
    use core::num::NonZero;

    #[derive(PartialEq, Eq)]
    pub struct Atom {
        index: NonZero<u32>, // private
    }

    pub const FOO_ATOM: Atom = Atom { index: unsafe { NonZero::new_unchecked(7) } };
}

macro_rules! atom {
    ("foo") => {
        atom::FOO_ATOM
    };
}

#[test]
fn test_match_nonzero_const_pattern() {
    match atom!("foo") {
        // Using as a pattern is supported by the compiler:
        atom!("foo") => {}
        _ => panic!("Expected the const item as a pattern to match."),
    }
}

#[test]
fn test_from_nonzero() {
    let nz = NonZero::new(1).unwrap();
    let num: u32 = nz.into();
    assert_eq!(num, 1u32);
}

#[test]
fn test_from_signed_nonzero() {
    let nz = NonZero::new(1).unwrap();
    let num: i32 = nz.into();
    assert_eq!(num, 1i32);
}

#[test]
fn test_from_str() {
    assert_eq!("123".parse::<NonZero<u8>>(), Ok(NonZero::new(123).unwrap()));
    assert_eq!(
        "0".parse::<NonZero<u8>>().err().map(|e| e.kind().clone()),
        Some(IntErrorKind::Zero)
    );
    assert_eq!(
        "-1".parse::<NonZero<u8>>().err().map(|e| e.kind().clone()),
        Some(IntErrorKind::InvalidDigit)
    );
    assert_eq!(
        "-129".parse::<NonZero<i8>>().err().map(|e| e.kind().clone()),
        Some(IntErrorKind::NegOverflow)
    );
    assert_eq!(
        "257".parse::<NonZero<u8>>().err().map(|e| e.kind().clone()),
        Some(IntErrorKind::PosOverflow)
    );
}

#[test]
fn test_nonzero_bitor() {
    let nz_alt = NonZero::new(0b1010_1010).unwrap();
    let nz_low = NonZero::new(0b0000_1111).unwrap();

    let both_nz: NonZero<u8> = nz_alt | nz_low;
    assert_eq!(both_nz.get(), 0b1010_1111);

    let rhs_int: NonZero<u8> = nz_low | 0b1100_0000u8;
    assert_eq!(rhs_int.get(), 0b1100_1111);

    let rhs_zero: NonZero<u8> = nz_alt | 0u8;
    assert_eq!(rhs_zero.get(), 0b1010_1010);

    let lhs_int: NonZero<u8> = 0b0110_0110u8 | nz_alt;
    assert_eq!(lhs_int.get(), 0b1110_1110);

    let lhs_zero: NonZero<u8> = 0u8 | nz_low;
    assert_eq!(lhs_zero.get(), 0b0000_1111);
}

#[test]
fn test_nonzero_bitor_assign() {
    let mut target = NonZero::<u8>::new(0b1010_1010).unwrap();

    target |= NonZero::new(0b0000_1111).unwrap();
    assert_eq!(target.get(), 0b1010_1111);

    target |= 0b0001_0000;
    assert_eq!(target.get(), 0b1011_1111);

    target |= 0;
    assert_eq!(target.get(), 0b1011_1111);
}

#[test]
fn test_nonzero_from_int_on_success() {
    assert_eq!(NonZero::<u8>::try_from(5), Ok(NonZero::new(5).unwrap()));
    assert_eq!(NonZero::<u32>::try_from(5), Ok(NonZero::new(5).unwrap()));

    assert_eq!(NonZero::<i8>::try_from(-5), Ok(NonZero::new(-5).unwrap()));
    assert_eq!(NonZero::<i32>::try_from(-5), Ok(NonZero::new(-5).unwrap()));
}

#[test]
fn test_nonzero_from_int_on_err() {
    assert!(NonZero::<u8>::try_from(0).is_err());
    assert!(NonZero::<u32>::try_from(0).is_err());

    assert!(NonZero::<i8>::try_from(0).is_err());
    assert!(NonZero::<i32>::try_from(0).is_err());
}

#[test]
fn nonzero_const() {
    // test that the methods of `NonZeroX>` are usable in a const context
    // Note: only tests NonZero<u8>

    const NONZERO_U8: NonZero<u8> = unsafe { NonZero::new_unchecked(5) };

    const GET: u8 = NONZERO_U8.get();
    assert_eq!(GET, 5);

    const ZERO: Option<NonZero<u8>> = NonZero::new(0);
    assert!(ZERO.is_none());

    const ONE: Option<NonZero<u8>> = NonZero::new(1);
    assert!(ONE.is_some());

    /* FIXME(#110395)
    const FROM_NONZERO_U8: u8 = u8::from(NONZERO_U8);
    assert_eq!(FROM_NONZERO_U8, 5);

    const NONZERO_CONVERT: NonZero<u32> = NonZero::<u32>::from(NONZERO_U8);
    assert_eq!(NONZERO_CONVERT.get(), 5);
    */
}

#[test]
fn nonzero_leading_zeros() {
    assert_eq!(NonZero::<u8>::new(1).unwrap().leading_zeros(), 7);
    assert_eq!(NonZero::<i8>::new(1).unwrap().leading_zeros(), 7);
    assert_eq!(NonZero::<u16>::new(1).unwrap().leading_zeros(), 15);
    assert_eq!(NonZero::<i16>::new(1).unwrap().leading_zeros(), 15);
    assert_eq!(NonZero::<u32>::new(1).unwrap().leading_zeros(), 31);
    assert_eq!(NonZero::<i32>::new(1).unwrap().leading_zeros(), 31);
    assert_eq!(NonZero::<u64>::new(1).unwrap().leading_zeros(), 63);
    assert_eq!(NonZero::<i64>::new(1).unwrap().leading_zeros(), 63);
    assert_eq!(NonZero::<u128>::new(1).unwrap().leading_zeros(), 127);
    assert_eq!(NonZero::<i128>::new(1).unwrap().leading_zeros(), 127);
    assert_eq!(NonZero::<usize>::new(1).unwrap().leading_zeros(), usize::BITS - 1);
    assert_eq!(NonZero::<isize>::new(1).unwrap().leading_zeros(), usize::BITS - 1);

    assert_eq!(NonZero::<u8>::new(u8::MAX >> 2).unwrap().leading_zeros(), 2);
    assert_eq!(NonZero::<i8>::new((u8::MAX >> 2) as i8).unwrap().leading_zeros(), 2);
    assert_eq!(NonZero::<u16>::new(u16::MAX >> 2).unwrap().leading_zeros(), 2);
    assert_eq!(NonZero::<i16>::new((u16::MAX >> 2) as i16).unwrap().leading_zeros(), 2);
    assert_eq!(NonZero::<u32>::new(u32::MAX >> 2).unwrap().leading_zeros(), 2);
    assert_eq!(NonZero::<i32>::new((u32::MAX >> 2) as i32).unwrap().leading_zeros(), 2);
    assert_eq!(NonZero::<u64>::new(u64::MAX >> 2).unwrap().leading_zeros(), 2);
    assert_eq!(NonZero::<i64>::new((u64::MAX >> 2) as i64).unwrap().leading_zeros(), 2);
    assert_eq!(NonZero::<u128>::new(u128::MAX >> 2).unwrap().leading_zeros(), 2);
    assert_eq!(NonZero::<i128>::new((u128::MAX >> 2) as i128).unwrap().leading_zeros(), 2);
    assert_eq!(NonZero::<usize>::new(usize::MAX >> 2).unwrap().leading_zeros(), 2);
    assert_eq!(NonZero::<isize>::new((usize::MAX >> 2) as isize).unwrap().leading_zeros(), 2);

    assert_eq!(NonZero::<u8>::new(u8::MAX).unwrap().leading_zeros(), 0);
    assert_eq!(NonZero::<i8>::new(-1i8).unwrap().leading_zeros(), 0);
    assert_eq!(NonZero::<u16>::new(u16::MAX).unwrap().leading_zeros(), 0);
    assert_eq!(NonZero::<i16>::new(-1i16).unwrap().leading_zeros(), 0);
    assert_eq!(NonZero::<u32>::new(u32::MAX).unwrap().leading_zeros(), 0);
    assert_eq!(NonZero::<i32>::new(-1i32).unwrap().leading_zeros(), 0);
    assert_eq!(NonZero::<u64>::new(u64::MAX).unwrap().leading_zeros(), 0);
    assert_eq!(NonZero::<i64>::new(-1i64).unwrap().leading_zeros(), 0);
    assert_eq!(NonZero::<u128>::new(u128::MAX).unwrap().leading_zeros(), 0);
    assert_eq!(NonZero::<i128>::new(-1i128).unwrap().leading_zeros(), 0);
    assert_eq!(NonZero::<usize>::new(usize::MAX).unwrap().leading_zeros(), 0);
    assert_eq!(NonZero::<isize>::new(-1isize).unwrap().leading_zeros(), 0);

    const LEADING_ZEROS: u32 = NonZero::<u16>::new(1).unwrap().leading_zeros();
    assert_eq!(LEADING_ZEROS, 15);
}

#[test]
fn nonzero_trailing_zeros() {
    assert_eq!(NonZero::<u8>::new(1).unwrap().trailing_zeros(), 0);
    assert_eq!(NonZero::<i8>::new(1).unwrap().trailing_zeros(), 0);
    assert_eq!(NonZero::<u16>::new(1).unwrap().trailing_zeros(), 0);
    assert_eq!(NonZero::<i16>::new(1).unwrap().trailing_zeros(), 0);
    assert_eq!(NonZero::<u32>::new(1).unwrap().trailing_zeros(), 0);
    assert_eq!(NonZero::<i32>::new(1).unwrap().trailing_zeros(), 0);
    assert_eq!(NonZero::<u64>::new(1).unwrap().trailing_zeros(), 0);
    assert_eq!(NonZero::<i64>::new(1).unwrap().trailing_zeros(), 0);
    assert_eq!(NonZero::<u128>::new(1).unwrap().trailing_zeros(), 0);
    assert_eq!(NonZero::<i128>::new(1).unwrap().trailing_zeros(), 0);
    assert_eq!(NonZero::<usize>::new(1).unwrap().trailing_zeros(), 0);
    assert_eq!(NonZero::<isize>::new(1).unwrap().trailing_zeros(), 0);

    assert_eq!(NonZero::<u8>::new(1 << 2).unwrap().trailing_zeros(), 2);
    assert_eq!(NonZero::<i8>::new(1 << 2).unwrap().trailing_zeros(), 2);
    assert_eq!(NonZero::<u16>::new(1 << 2).unwrap().trailing_zeros(), 2);
    assert_eq!(NonZero::<i16>::new(1 << 2).unwrap().trailing_zeros(), 2);
    assert_eq!(NonZero::<u32>::new(1 << 2).unwrap().trailing_zeros(), 2);
    assert_eq!(NonZero::<i32>::new(1 << 2).unwrap().trailing_zeros(), 2);
    assert_eq!(NonZero::<u64>::new(1 << 2).unwrap().trailing_zeros(), 2);
    assert_eq!(NonZero::<i64>::new(1 << 2).unwrap().trailing_zeros(), 2);
    assert_eq!(NonZero::<u128>::new(1 << 2).unwrap().trailing_zeros(), 2);
    assert_eq!(NonZero::<i128>::new(1 << 2).unwrap().trailing_zeros(), 2);
    assert_eq!(NonZero::<usize>::new(1 << 2).unwrap().trailing_zeros(), 2);
    assert_eq!(NonZero::<isize>::new(1 << 2).unwrap().trailing_zeros(), 2);

    assert_eq!(NonZero::<u8>::new(1 << 7).unwrap().trailing_zeros(), 7);
    assert_eq!(NonZero::<i8>::new(1 << 7).unwrap().trailing_zeros(), 7);
    assert_eq!(NonZero::<u16>::new(1 << 15).unwrap().trailing_zeros(), 15);
    assert_eq!(NonZero::<i16>::new(1 << 15).unwrap().trailing_zeros(), 15);
    assert_eq!(NonZero::<u32>::new(1 << 31).unwrap().trailing_zeros(), 31);
    assert_eq!(NonZero::<i32>::new(1 << 31).unwrap().trailing_zeros(), 31);
    assert_eq!(NonZero::<u64>::new(1 << 63).unwrap().trailing_zeros(), 63);
    assert_eq!(NonZero::<i64>::new(1 << 63).unwrap().trailing_zeros(), 63);
    assert_eq!(NonZero::<u128>::new(1 << 127).unwrap().trailing_zeros(), 127);
    assert_eq!(NonZero::<i128>::new(1 << 127).unwrap().trailing_zeros(), 127);

    assert_eq!(
        NonZero::<usize>::new(1 << (usize::BITS - 1)).unwrap().trailing_zeros(),
        usize::BITS - 1
    );
    assert_eq!(
        NonZero::<isize>::new(1 << (usize::BITS - 1)).unwrap().trailing_zeros(),
        usize::BITS - 1
    );

    const TRAILING_ZEROS: u32 = NonZero::<u16>::new(1 << 2).unwrap().trailing_zeros();
    assert_eq!(TRAILING_ZEROS, 2);
}

#[test]
fn test_nonzero_isolate_most_significant_one() {
    // Signed most significant one
    macro_rules! nonzero_int_impl {
        ($($T:ty),+) => {
            $(
                {
                    const BITS: $T = -1;
                    const MOST_SIG_ONE: $T = 1 << (<$T>::BITS - 1);

                    // Right shift the most significant one through each
                    // bit position, starting with all bits set
                    let mut i = 0;
                    while i < <$T>::BITS {
                        assert_eq!(
                            NonZero::<$T>::new(BITS >> i).unwrap().isolate_most_significant_one(),
                            NonZero::<$T>::new(MOST_SIG_ONE >> i).unwrap().isolate_most_significant_one()
                        );
                        i += 1;
                    }
                }
            )+
        };
    }

    // Unsigned most significant one
    macro_rules! nonzero_uint_impl {
        ($($T:ty),+) => {
            $(
                {
                    const BITS: $T = <$T>::MAX;
                    const MOST_SIG_ONE: $T = 1 << (<$T>::BITS - 1);

                    let mut i = 0;
                    while i < <$T>::BITS {
                        assert_eq!(
                            NonZero::<$T>::new(BITS >> i).unwrap().isolate_most_significant_one(),
                            NonZero::<$T>::new(MOST_SIG_ONE >> i).unwrap().isolate_most_significant_one(),
                        );
                        i += 1;
                    }
                }
            )+
        };
    }

    nonzero_int_impl!(i8, i16, i32, i64, i128, isize);
    nonzero_uint_impl!(u8, u16, u32, u64, u128, usize);
}

#[test]
fn test_nonzero_isolate_least_significant_one() {
    // Signed least significant one
    macro_rules! nonzero_int_impl {
        ($($T:ty),+) => {
            $(
                {
                    const BITS: $T = -1;
                    const LEAST_SIG_ONE: $T = 1;

                    // Left shift the least significant one through each
                    // bit position, starting with all bits set
                    let mut i = 0;
                    while i < <$T>::BITS {
                        assert_eq!(
                            NonZero::<$T>::new(BITS << i).unwrap().isolate_least_significant_one(),
                            NonZero::<$T>::new(LEAST_SIG_ONE << i).unwrap().isolate_least_significant_one()
                        );
                        i += 1;
                    }
                }
            )+
        };
    }

    // Unsigned least significant one
    macro_rules! nonzero_uint_impl {
        ($($T:ty),+) => {
            $(
                {
                    const BITS: $T = <$T>::MAX;
                    const LEAST_SIG_ONE: $T = 1;

                    let mut i = 0;
                    while i < <$T>::BITS {
                        assert_eq!(
                            NonZero::<$T>::new(BITS << i).unwrap().isolate_least_significant_one(),
                            NonZero::<$T>::new(LEAST_SIG_ONE << i).unwrap().isolate_least_significant_one(),
                        );
                        i += 1;
                    }
                }
            )+
        };
    }

    nonzero_int_impl!(i8, i16, i32, i64, i128, isize);
    nonzero_uint_impl!(u8, u16, u32, u64, u128, usize);
}

#[test]
fn test_nonzero_uint_div() {
    let nz = NonZero::new(1).unwrap();

    let x: u32 = 42u32 / nz;
    assert_eq!(x, 42u32);
}

#[test]
fn test_nonzero_uint_rem() {
    let nz = NonZero::new(10).unwrap();

    let x: u32 = 42u32 % nz;
    assert_eq!(x, 2u32);
}

#[test]
fn test_signed_nonzero_neg() {
    assert_eq!((-NonZero::<i8>::new(1).unwrap()).get(), -1);
    assert_eq!((-NonZero::<i8>::new(-1).unwrap()).get(), 1);

    assert_eq!((-NonZero::<i16>::new(1).unwrap()).get(), -1);
    assert_eq!((-NonZero::<i16>::new(-1).unwrap()).get(), 1);

    assert_eq!((-NonZero::<i32>::new(1).unwrap()).get(), -1);
    assert_eq!((-NonZero::<i32>::new(-1).unwrap()).get(), 1);

    assert_eq!((-NonZero::<i64>::new(1).unwrap()).get(), -1);
    assert_eq!((-NonZero::<i64>::new(-1).unwrap()).get(), 1);

    assert_eq!((-NonZero::<i128>::new(1).unwrap()).get(), -1);
    assert_eq!((-NonZero::<i128>::new(-1).unwrap()).get(), 1);
}

#[test]
fn test_nonzero_fmt() {
    let i = format!("{0}, {0:?}, {0:x}, {0:X}, {0:#x}, {0:#X}, {0:o}, {0:b}, {0:e}, {0:E}", 42);
    let nz = format!(
        "{0}, {0:?}, {0:x}, {0:X}, {0:#x}, {0:#X}, {0:o}, {0:b}, {0:e}, {0:E}",
        NonZero::new(42).unwrap()
    );

    assert_eq!(i, nz);
}
