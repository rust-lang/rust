use crate::core_arch::{aarch64::neon::*, arm_shared::*, simd::*};
use std::{i16, i32, i8, mem::transmute, u16, u32, u8, vec::Vec};

macro_rules! V_u64 {
    () => {
        vec![
            0x0000000000000000u64,
            0x0101010101010101u64,
            0x0202020202020202u64,
            0x0F0F0F0F0F0F0F0Fu64,
            0x8080808080808080u64,
            0xF0F0F0F0F0F0F0F0u64,
            0xFFFFFFFFFFFFFFFFu64,
        ]
    };
}

macro_rules! V_f64 {
    () => {
        vec![
            0.0f64,
            1.0f64,
            -1.0f64,
            1.2f64,
            2.4f64,
            std::f64::MAX,
            std::f64::MIN,
            std::f64::INFINITY,
            std::f64::NEG_INFINITY,
            std::f64::NAN,
        ]
    };
}

macro_rules! to64 {
    ($t : ident) => {
        |v: $t| -> u64 { transmute(v) }
    };
}

macro_rules! to128 {
    ($t : ident) => {
        |v: $t| -> u128 { transmute(v) }
    };
}

pub(crate) fn test<T, U, V, W, X>(
    vals: Vec<T>,
    fill1: fn(T) -> V,
    fill2: fn(U) -> W,
    cast: fn(W) -> X,
    test_fun: fn(V, V) -> W,
    verify_fun: fn(T, T) -> U,
) where
    T: Copy + core::fmt::Debug,
    U: Copy + core::fmt::Debug + std::cmp::PartialEq,
    V: Copy + core::fmt::Debug,
    W: Copy + core::fmt::Debug,
    X: Copy + core::fmt::Debug + std::cmp::PartialEq,
{
    let pairs = vals.iter().zip(vals.iter());

    for (i, j) in pairs {
        let a: V = fill1(*i);
        let b: V = fill1(*j);

        let actual_pre: W = test_fun(a, b);
        let expected_pre: W = fill2(verify_fun(*i, *j));

        let actual: X = cast(actual_pre);
        let expected: X = cast(expected_pre);

        assert_eq!(
            actual, expected,
            "[{:?}:{:?}] :\nf({:?}, {:?}) = {:?}\ng({:?}, {:?}) = {:?}\n",
            *i, *j, &a, &b, actual_pre, &a, &b, expected_pre
        );
    }
}

macro_rules! gen_test_fn {
    ($n: ident, $t: ident, $u: ident, $v: ident, $w: ident, $x: ident, $vals: expr, $fill1: expr, $fill2: expr, $cast: expr) => {
        pub(crate) fn $n(test_fun: fn($v, $v) -> $w, verify_fun: fn($t, $t) -> $u) {
            unsafe {
                test::<$t, $u, $v, $w, $x>($vals, $fill1, $fill2, $cast, test_fun, verify_fun)
            };
        }
    };
}

macro_rules! gen_fill_fn {
    ($id: ident, $el_width: expr, $num_els: expr, $in_t : ident, $out_t: ident, $cmp_t: ident) => {
        pub(crate) fn $id(val: $in_t) -> $out_t {
            let initial: [$in_t; $num_els] = [val; $num_els];
            let result: $cmp_t = unsafe { transmute(initial) };
            let result_out: $out_t = unsafe { transmute(result) };

            // println!("FILL: {:016x} as {} x {}: {:016x}", val.reverse_bits(), $el_width, $num_els, (result as u64).reverse_bits());

            result_out
        }
    };
}

gen_fill_fn!(fill_u64, 64, 1, u64, uint64x1_t, u64);
gen_fill_fn!(fillq_u64, 64, 2, u64, uint64x2_t, u128);
gen_fill_fn!(fill_f64, 64, 1, f64, float64x1_t, u64);
gen_fill_fn!(fillq_f64, 64, 2, f64, float64x2_t, u128);
gen_fill_fn!(fill_p64, 64, 1, u64, poly64x1_t, u64);
gen_fill_fn!(fillq_p64, 64, 2, u64, poly64x2_t, u128);

gen_test_fn!(
    test_ari_f64,
    f64,
    f64,
    float64x1_t,
    float64x1_t,
    u64,
    V_f64!(),
    fill_f64,
    fill_f64,
    to64!(float64x1_t)
);
gen_test_fn!(
    test_cmp_f64,
    f64,
    u64,
    float64x1_t,
    uint64x1_t,
    u64,
    V_f64!(),
    fill_f64,
    fill_u64,
    to64!(uint64x1_t)
);
gen_test_fn!(
    testq_ari_f64,
    f64,
    f64,
    float64x2_t,
    float64x2_t,
    u128,
    V_f64!(),
    fillq_f64,
    fillq_f64,
    to128!(float64x2_t)
);
gen_test_fn!(
    testq_cmp_f64,
    f64,
    u64,
    float64x2_t,
    uint64x2_t,
    u128,
    V_f64!(),
    fillq_f64,
    fillq_u64,
    to128!(uint64x2_t)
);

gen_test_fn!(
    test_cmp_p64,
    u64,
    u64,
    poly64x1_t,
    uint64x1_t,
    u64,
    V_u64!(),
    fill_p64,
    fill_u64,
    to64!(uint64x1_t)
);
gen_test_fn!(
    testq_cmp_p64,
    u64,
    u64,
    poly64x2_t,
    uint64x2_t,
    u128,
    V_u64!(),
    fillq_p64,
    fillq_u64,
    to128!(uint64x2_t)
);
