use test::{Bencher, black_box};

#[bench]
fn bench_0(b: &mut Bencher) {
    b.iter(|| black_box("0.0").parse::<f64>());
}

#[bench]
fn bench_42(b: &mut Bencher) {
    b.iter(|| black_box("42").parse::<f64>());
}

#[bench]
fn bench_huge_int(b: &mut Bencher) {
    // 2^128 - 1
    b.iter(|| black_box("170141183460469231731687303715884105727").parse::<f64>());
}

#[bench]
fn bench_short_decimal(b: &mut Bencher) {
    b.iter(|| black_box("1234.5678").parse::<f64>());
}

#[bench]
fn bench_pi_long(b: &mut Bencher) {
    b.iter(|| black_box("3.14159265358979323846264338327950288").parse::<f64>());
}

#[bench]
fn bench_pi_short(b: &mut Bencher) {
    b.iter(|| black_box("3.141592653589793").parse::<f64>())
}

#[bench]
fn bench_1e150(b: &mut Bencher) {
    b.iter(|| black_box("1e150").parse::<f64>());
}

#[bench]
fn bench_long_decimal_and_exp(b: &mut Bencher) {
    b.iter(|| black_box("727501488517303786137132964064381141071e-123").parse::<f64>());
}

#[bench]
fn bench_min_subnormal(b: &mut Bencher) {
    b.iter(|| black_box("5e-324").parse::<f64>());
}

#[bench]
fn bench_min_normal(b: &mut Bencher) {
    b.iter(|| black_box("2.2250738585072014e-308").parse::<f64>());
}

#[bench]
fn bench_max(b: &mut Bencher) {
    b.iter(|| black_box("1.7976931348623157e308").parse::<f64>());
}
