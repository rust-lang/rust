use std::panic;

use quickcheck::TestResult;

macro_rules! absv_i2 {
    ($intrinsic:ident: $ty:ident) => {
        #[test]
        fn $intrinsic() {
            assert!(panic::catch_unwind(|| ::$intrinsic(::std::$ty::MIN)).is_err());
            assert_eq!(::$intrinsic(::std::$ty::MIN + 1), ::std::$ty::MAX);
            assert_eq!(::$intrinsic(::std::$ty::MIN + 2), ::std::$ty::MAX - 1);
            assert_eq!(::$intrinsic(-1), 1);
            assert_eq!(::$intrinsic(-2), 2);
            assert_eq!(::$intrinsic(0), 0);
            assert_eq!(::$intrinsic(1), 1);
            assert_eq!(::$intrinsic(2), 2);
            assert_eq!(::$intrinsic(2), 2);
            assert_eq!(::$intrinsic(::std::$ty::MAX - 1), ::std::$ty::MAX - 1);
            assert_eq!(::$intrinsic(::std::$ty::MAX), ::std::$ty::MAX);
        }
    }
}

absv_i2!(__absvsi2: i32);
absv_i2!(__absvdi2: i64);
// TODO(rust-lang/35118)?
// absv_i2!(__absvti2: i128);

quickcheck! {
    fn udivmoddi4(n: (u32, u32), d: (u32, u32)) -> TestResult {
        let n = ::U64 { low: n.0, high: n.1 }[..];
        let d = ::U64 { low: d.0, high: d.1 }[..];

        if d == 0 {
            TestResult::discard()
        } else {
            let mut r = 0;
            let q = ::div::__udivmoddi4(n, d, Some(&mut r));

            TestResult::from_bool(q * d + r == n)
        }
    }
}

quickcheck! {
    fn udivmodsi4(n: u32, d: u32) -> TestResult {
        if d == 0 {
            TestResult::discard()
        } else {
            let mut r = 0;
            let q = ::div::__udivmodsi4(n, d, Some(&mut r));

            TestResult::from_bool(q * d + r == n)
        }
    }
}
