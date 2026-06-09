use rand::Rng;
use test::{Bencher, black_box};

macro_rules! int_log10_bench {
    ($t:ty, $predictable:ident, $random:ident, $random_small:ident) => {
        #[bench]
        fn $predictable(bench: &mut Bencher) {
            bench.iter(|| {
                for n in 0..(<$t>::BITS / 8) {
                    for i in 1..=(100 as $t) {
                        let x = black_box(i << (n * 8));
                        black_box(x.ilog10());
                    }
                }
            });
        }

        #[bench]
        fn $random(bench: &mut Bencher) {
            let mut rng = crate::bench_rng();
            /* Exponentially distributed random numbers from the whole range of the type.  */
            let numbers: Vec<$t> = (0..256)
                .map(|_| {
                    let x = rng.random::<$t>() >> rng.random_range(0..<$t>::BITS);
                    if x != 0 { x } else { 1 }
                })
                .collect();
            bench.iter(|| {
                for x in &numbers {
                    black_box(black_box(x).ilog10());
                }
            });
        }

        #[bench]
        fn $random_small(bench: &mut Bencher) {
            let mut rng = crate::bench_rng();
            /* Exponentially distributed random numbers from the range 0..256.  */
            let numbers: Vec<$t> = (0..256)
                .map(|_| {
                    let x = (rng.random::<u8>() >> rng.random_range(0..u8::BITS)) as $t;
                    if x != 0 { x } else { 1 }
                })
                .collect();
            bench.iter(|| {
                for x in &numbers {
                    black_box(black_box(x).ilog10());
                }
            });
        }
    };
}

int_log10_bench! {u8, u8_log10_predictable, u8_log10_random, u8_log10_random_small}
int_log10_bench! {u16, u16_log10_predictable, u16_log10_random, u16_log10_random_small}
int_log10_bench! {u32, u32_log10_predictable, u32_log10_random, u32_log10_random_small}
int_log10_bench! {u64, u64_log10_predictable, u64_log10_random, u64_log10_random_small}
int_log10_bench! {u128, u128_log10_predictable, u128_log10_random, u128_log10_random_small}

macro_rules! int_log_bench {
    ($t:ty, $random:ident, $random_small:ident, $geometric:ident) => {
        #[bench]
        fn $random(bench: &mut Bencher) {
            let mut rng = crate::bench_rng();
            /* Exponentially distributed random numbers from the whole range of the type.  */
            let numbers: Vec<$t> = (0..256)
                .map(|_| {
                    let x = rng.random::<$t>() >> rng.random_range(0..<$t>::BITS);
                    if x >= 2 { x } else { 2 }
                })
                .collect();
            bench.iter(|| {
                for &b in &numbers {
                    for &x in &numbers {
                        black_box(black_box(x).ilog(b));
                    }
                }
            });
        }

        #[bench]
        fn $random_small(bench: &mut Bencher) {
            let mut rng = crate::bench_rng();
            /* Exponentially distributed random numbers from the range 0..256.  */
            let numbers: Vec<$t> = (0..256)
                .map(|_| {
                    let x = (rng.random::<u8>() >> rng.random_range(0..u8::BITS)) as $t;
                    if x >= 2 { x } else { 2 }
                })
                .collect();
            bench.iter(|| {
                for &b in &numbers {
                    for &x in &numbers {
                        black_box(black_box(x).ilog(b));
                    }
                }
            });
        }

        #[bench]
        fn $geometric(bench: &mut Bencher) {
            let bases: [$t; 16] = [2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65];
            let base_and_numbers: Vec<($t, Vec<$t>)> = bases
                .iter()
                .map(|&b| {
                    let numbers = (0..=<$t>::MAX.ilog(b)).map(|exp| b.pow(exp)).collect();
                    (b, numbers)
                })
                .collect();
            bench.iter(|| {
                for (b, numbers) in &base_and_numbers {
                    for &x in numbers {
                        black_box(black_box(x).ilog(black_box(*b)));
                    }
                }
            });
        }
    };
}

int_log_bench! {u8, u8_log_random, u8_log_random_small, u8_log_geometric}
int_log_bench! {u16, u16_log_random, u16_log_random_small, u16_log_geometric}
int_log_bench! {u32, u32_log_random, u32_log_random_small, u32_log_geometric}
int_log_bench! {u64, u64_log_random, u64_log_random_small, u64_log_geometric}
int_log_bench! {u128, u128_log_random, u128_log_random_small, u128_log_geometric}
