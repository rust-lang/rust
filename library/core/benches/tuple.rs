use rand::prelude::*;
use test::{black_box, Bencher};

#[bench]
fn bench_tuple_comparison(b: &mut Bencher) {
    let mut rng = black_box(super::bench_rng());

    let data = black_box([
        ("core::iter::adapters::Chain", 123_usize),
        ("core::iter::adapters::Clone", 456_usize),
        ("core::iter::adapters::Copie", 789_usize),
        ("core::iter::adapters::Cycle", 123_usize),
        ("core::iter::adapters::Flatt", 456_usize),
        ("core::iter::adapters::TakeN", 789_usize),
    ]);

    b.iter(|| {
        let x = data.choose(&mut rng).unwrap();
        let y = data.choose(&mut rng).unwrap();
        [x < y, x <= y, x > y, x >= y]
    });
}
