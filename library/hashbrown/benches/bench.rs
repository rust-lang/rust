#![expect(missing_docs)] // criterion_group! generates a public bench entrypoint

use criterion::{criterion_group, criterion_main};

mod general_ops;
mod insert_unique_unchecked;
mod set_ops;
mod with_capacity;

criterion_group!(
    benches,
    general_ops::register_benches,
    insert_unique_unchecked::register_benches,
    set_ops::register_benches,
    with_capacity::register_benches
);
criterion_main!(benches);
