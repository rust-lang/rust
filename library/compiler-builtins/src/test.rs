use std::{mem, panic};

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
    fn udivmoddi4(a: (u32, u32), b: (u32, u32)) -> TestResult {
        let (a, b) = unsafe {
            (mem::transmute(a), mem::transmute(b))
        };

        if b == 0 {
            TestResult::discard()
        } else {
            let mut r = 0;
            let q = ::__udivmoddi4(a, b, &mut r);

            TestResult::from_bool(q * b + r == a)
        }
    }
}
