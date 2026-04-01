//! Compare `insert` and `insert_unique_unchecked` operations performance.
#![expect(missing_docs)] // criterion_group! generates a public bench entrypoint

use criterion::{Criterion, criterion_group, criterion_main};
use hashbrown::HashMap;
use std::hint::black_box;

fn register_benches(c: &mut Criterion) {
    let keys: Vec<String> = (0..1000).map(|i| format!("xxxx{i}yyyy")).collect();

    c.bench_function("insert", |b| {
        let mut m = HashMap::with_capacity(1000);
        b.iter(|| {
            m.clear();
            for k in &keys {
                m.insert(k, k);
            }
            black_box(m.len())
        });
    });

    c.bench_function("insert_unique_unchecked", |b| {
        let mut m = HashMap::with_capacity(1000);
        b.iter(|| {
            m.clear();
            for k in &keys {
                unsafe {
                    m.insert_unique_unchecked(k, k);
                }
            }
            black_box(m.len())
        });
    });
}

criterion_group!(benches, register_benches);
criterion_main!(benches);
