use crate::fmt::Debug;

#[track_caller]
#[allow(unused)]
pub(crate) fn assert_eq_rt<T: PartialEq + Debug>(a: &T, b: &T) {
    std::assert_eq!(a, b)
}

#[allow(unused)]
macro_rules! assert_eq_const {
    ($a:expr, $b:expr $(,)?) => {{
        #[inline(always)]
        #[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
        const fn assert_eq_ct<T: [const] PartialEq>(a: &T, b: &T) {
            assert!(a == b, concat!("`", stringify!($a), "` != `", stringify!($b), "`"));
        }

        $crate::intrinsics::const_eval_select((&$a, &$b), assert_eq_ct, $crate::core_arch::test::assert_eq_rt);
    }};
    ($a:expr, $b:expr, $($t:tt)+) => {
        ::std::assert_eq!($a, $b, $($t)+)
    };
}

pub(crate) use assert_eq_const;
