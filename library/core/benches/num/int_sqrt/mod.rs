use rand::Rng;
use test::{Bencher, black_box};

macro_rules! int_sqrt_bench {
    ($t:ty, $predictable:ident, $random:ident, $random_small:ident, $random_uniform:ident) => {
        #[bench]
        fn $predictable(bench: &mut Bencher) {
            bench.iter(|| {
                for n in 0..(<$t>::BITS / 8) {
                    for i in 1..=(100 as $t) {
                        let x = black_box(i << (n * 8));
                        black_box(x.isqrt());
                    }
                }
            });
        }

        #[bench]
        fn $random(bench: &mut Bencher) {
            let mut rng = crate::bench_rng();
            /* Exponentially distributed random numbers from the whole range of the type.  */
            let numbers: Vec<$t> =
                (0..256).map(|_| rng.gen::<$t>() >> rng.gen_range(0..<$t>::BITS)).collect();
            bench.iter(|| {
                for x in &numbers {
                    black_box(black_box(x).isqrt());
                }
            });
        }

        #[bench]
        fn $random_small(bench: &mut Bencher) {
            let mut rng = crate::bench_rng();
            /* Exponentially distributed random numbers from the range 0..256.  */
            let numbers: Vec<$t> =
                (0..256).map(|_| (rng.gen::<u8>() >> rng.gen_range(0..u8::BITS)) as $t).collect();
            bench.iter(|| {
                for x in &numbers {
                    black_box(black_box(x).isqrt());
                }
            });
        }

        #[bench]
        fn $random_uniform(bench: &mut Bencher) {
            let mut rng = crate::bench_rng();
            /* Exponentially distributed random numbers from the whole range of the type.  */
            let numbers: Vec<$t> = (0..256).map(|_| rng.gen::<$t>()).collect();
            bench.iter(|| {
                for x in &numbers {
                    black_box(black_box(x).isqrt());
                }
            });
        }
    };
}

int_sqrt_bench! {u8, u8_sqrt_predictable, u8_sqrt_random, u8_sqrt_random_small, u8_sqrt_uniform}
int_sqrt_bench! {u16, u16_sqrt_predictable, u16_sqrt_random, u16_sqrt_random_small, u16_sqrt_uniform}
int_sqrt_bench! {u32, u32_sqrt_predictable, u32_sqrt_random, u32_sqrt_random_small, u32_sqrt_uniform}
int_sqrt_bench! {u64, u64_sqrt_predictable, u64_sqrt_random, u64_sqrt_random_small, u64_sqrt_uniform}
int_sqrt_bench! {u128, u128_sqrt_predictable, u128_sqrt_random, u128_sqrt_random_small, u128_sqrt_uniform}
