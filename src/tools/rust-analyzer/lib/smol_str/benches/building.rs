#[macro_use]
extern crate criterion;
extern crate smol_str;

use criterion::{Criterion, ParameterizedBenchmark, Throughput};
use smol_str::SmolStr;

fn from_str_iter(c: &mut Criterion) {
    use std::iter::FromIterator;

    const SIZES: &[usize] = &[0, 5, 10, 15, 20, 2 << 4, 2 << 5, 2 << 6, 2 << 7, 2 << 8];

    fn test_data(input: &str, size: usize) -> Vec<&str> {
        std::iter::repeat(input).take(size / input.len()).collect()
    }

    c.bench(
        "FromIterator",
        ParameterizedBenchmark::new(
            "SmolStr, one byte elements",
            |b, &&size| {
                let src = test_data("x", size);
                b.iter(|| SmolStr::from_iter(src.iter().cloned()).len())
            },
            SIZES,
        )
        .with_function("SmolStr, five byte elements", |b, &&size| {
            let src = test_data("helloo", size);
            b.iter(|| SmolStr::from_iter(src.iter().cloned()).len())
        })
        .with_function("String, one byte elements", |b, &&size| {
            let src = test_data("x", size);
            b.iter(|| String::from_iter(src.iter().cloned()).len())
        })
        .with_function("String, five byte elements", |b, &&size| {
            let src = test_data("hello", size);
            b.iter(|| String::from_iter(src.iter().cloned()).len())
        })
        .throughput(|elems| Throughput::Bytes(**elems as u32)),
    );
}

criterion_group!(benches, from_str_iter);
criterion_main!(benches);
