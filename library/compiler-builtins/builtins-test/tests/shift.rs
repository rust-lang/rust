use builtins_test::*;

macro_rules! shift {
    ($($i:ty, $fn_std:ident, $fn_builtins:ident);*;) => {
        $(
            #[test]
            fn $fn_builtins() {
                use compiler_builtins::int::shift::$fn_builtins;

                fuzz_shift(|x: $i, s: u32| {
                    let tmp0: $i = x.$fn_std(s);
                    let tmp1: $i = $fn_builtins(x, s);
                    if tmp0 != tmp1 {
                        panic!(
                            "{}({}, {}): std: {}, builtins: {}",
                            stringify!($fn_builtins), x, s, tmp0, tmp1
                        );
                    }
                });
            }
        )*
    };
}

shift! {
    u32, wrapping_shl, __ashlsi3;
    u64, wrapping_shl, __ashldi3;
    u128, wrapping_shl, __ashlti3;
    i32, wrapping_shr, __ashrsi3;
    i64, wrapping_shr, __ashrdi3;
    i128, wrapping_shr, __ashrti3;
    u32, wrapping_shr, __lshrsi3;
    u64, wrapping_shr, __lshrdi3;
    u128, wrapping_shr, __lshrti3;
}
