use std::panic;

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
