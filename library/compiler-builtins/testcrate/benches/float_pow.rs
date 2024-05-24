use compiler_builtins::float::pow;
use criterion::{criterion_group, criterion_main, Criterion};
use testcrate::float_bench;

float_bench! {
    name: powi_f32,
    sig: (a: f32, b: i32) -> f32,
    crate_fn: pow::__powisf2,
    sys_fn: __powisf2,
    sys_available: all(),
    asm: [],
}

float_bench! {
    name: powi_f64,
    sig: (a: f64, b: i32) -> f64,
    crate_fn: pow::__powidf2,
    sys_fn: __powidf2,
    sys_available: all(),
    asm: [],
}

criterion_group!(float_add, powi_f32, powi_f64);
criterion_main!(float_add);
