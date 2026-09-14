use std::sync::LazyLock;

// All (negative) integers which are at or near a power of two to test
// boundary conditions. We use strings so we can convert to any type using
// parsing, while still being able to use the *position* in ORDERED_VALS for
// comparisons.
static ORDERED_VALS: LazyLock<Vec<String>> = LazyLock::new(|| {
    let mut pos_int_vals = Vec::new();
    for exp in 0..=127 {
        let val = 1_u128 << exp;
        pos_int_vals.push(val.saturating_sub(2));
        pos_int_vals.push(val.saturating_sub(1));
        pos_int_vals.push(val);
        pos_int_vals.push(val.saturating_add(1));
        pos_int_vals.push(val.saturating_add(2));
    }
    pos_int_vals.sort();
    pos_int_vals.dedup();

    let mut pos_str_vals: Vec<_> = pos_int_vals.iter().map(|i| i.to_string()).collect();

    // These are manual because the upper ones overflow even u128.
    pos_str_vals.push("340282366920938463463374607431768211454".to_owned()); // 2**128 - 2
    pos_str_vals.push("340282366920938463463374607431768211455".to_owned()); // 2**128 - 1
    pos_str_vals.push("340282366920938463463374607431768211456".to_owned()); // 2**128
    pos_str_vals.push("340282366920938463463374607431768211457".to_owned()); // 2**128 + 1
    pos_str_vals.push("340282366920938463463374607431768211458".to_owned()); // 2**128 + 2

    let mut out = Vec::new();
    for val in pos_str_vals[1..].iter().rev() {
        out.push(format!("-{val}"));
    }
    out.extend(pos_str_vals);
    out
});

macro_rules! make_checked_cast_test {
    ($Src:ident as [$($Dst:ident),*]) => {$(
        #[test]
        #[allow(non_snake_case)]
        fn ${concat(test_checked_cast_, $Src, _to_, $Dst)}() {
            for val in ORDERED_VALS.iter() {
                if let Some(src) = val.parse::<$Src>().ok() {
                    let dst: Option<$Dst> = val.parse().ok();
                    assert_eq!(src.checked_cast::<$Dst>(), dst);
                }
            }
        }
    )*}
}

macro_rules! make_bounded_cast_test {
    (|$src:ident| $raw:expr, $Src:ident as [$($Dst:ident),*]) => {$(
        #[test]
        #[allow(non_snake_case)]
        fn ${concat(test_bounded_cast_, $Src, _to_, $Dst)}() {
            let ord_idx = |s| ORDERED_VALS.iter().position(|v| *v == s).unwrap();
            let dst_min_idx = ord_idx(<$Dst>::MIN.to_string());
            let dst_max_idx = ord_idx(<$Dst>::MAX.to_string());
            for (val_idx, val) in ORDERED_VALS.iter().enumerate() {
                if let Some($src) = val.parse::<$Src>().ok() {
                    let dst: Option<$Dst> = val.parse().ok();

                    assert_eq!($src.wrapping_cast::<$Dst>(), $raw as $Dst);

                    if val_idx > dst_max_idx {
                        assert_eq!($src.saturating_cast::<$Dst>(), <$Dst>::MAX);
                    } else if val_idx < dst_min_idx {
                        assert_eq!($src.saturating_cast::<$Dst>(), <$Dst>::MIN);
                    } else {
                        assert_eq!($src.saturating_cast::<$Dst>(), dst.unwrap());
                    }
                }
            }
        }
    )*}
}

macro_rules! make_tests_for_src {
    (|$src:ident| $raw:expr, [$($Src:ident),*]) => {$(
        make_checked_cast_test!(             $Src as [u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize]);
        make_bounded_cast_test!(|$src| $raw, $Src as [u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize]);

        // NonZero types are not (yet) implemented.
        // make_checked_cast_test!($Src as [
        //     NonZeroU8, NonZeroU16, NonZeroU32, NonZeroU64, NonZeroU128, NonZeroUsize,
        //     NonZeroI8, NonZeroI16, NonZeroI32, NonZeroI64, NonZeroI128, NonZeroIsize
        // ]);
    )*}
}

make_tests_for_src!(|x| x, [u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize]);

// NonZero types are not (yet) implemented.
// make_tests_for_src!(
//     |x| x.get(),
//     [
//         NonZeroU8, NonZeroU16, NonZeroU32, NonZeroU64, NonZeroU128, NonZeroUsize,
//         NonZeroI8, NonZeroI16, NonZeroI32, NonZeroI64, NonZeroI128, NonZeroIsize
//     ]
// );
