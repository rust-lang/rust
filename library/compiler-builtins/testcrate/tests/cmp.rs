use testcrate::*;

macro_rules! cmp {
    ($x:ident, $y:ident, $($unordered_val:expr, $fn:ident);*;) => {
        $(
            let cmp0 = if $x.is_nan() || $y.is_nan() {
                $unordered_val
            } else if $x < $y {
                -1
            } else if $x == $y {
                0
            } else {
                1
            };
            let cmp1 = $fn($x, $y);
            if cmp0 != cmp1 {
                panic!("{}({}, {}): std: {}, builtins: {}", stringify!($fn_builtins), $x, $y, cmp0, cmp1);
            }
        )*
    };
}

#[test]
fn float_comparisons() {
    use compiler_builtins::float::cmp::{
        __eqdf2, __eqsf2, __gedf2, __gesf2, __gtdf2, __gtsf2, __ledf2, __lesf2, __ltdf2, __ltsf2,
        __nedf2, __nesf2, __unorddf2, __unordsf2,
    };

    fuzz_float_2(N, |x: f32, y: f32| {
        assert_eq!(__unordsf2(x, y) != 0, x.is_nan() || y.is_nan());
        cmp!(x, y,
            1, __ltsf2;
            1, __lesf2;
            1, __eqsf2;
            -1, __gesf2;
            -1, __gtsf2;
            1, __nesf2;
        );
    });
    fuzz_float_2(N, |x: f64, y: f64| {
        assert_eq!(__unorddf2(x, y) != 0, x.is_nan() || y.is_nan());
        cmp!(x, y,
            1, __ltdf2;
            1, __ledf2;
            1, __eqdf2;
            -1, __gedf2;
            -1, __gtdf2;
            1, __nedf2;
        );
    });
}
