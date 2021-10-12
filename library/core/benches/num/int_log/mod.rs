use rand::Rng;
use test::{black_box, Bencher};

macro_rules! int_log_bench {
    ($t:ty, $predictable:ident, $random:ident, $random_small:ident) => {
        #[bench]
        fn $predictable(bench: &mut Bencher) {
            bench.iter(|| {
                for n in 0..(<$t>::BITS / 8) {
                    for i in 1..=(100 as $t) {
                        let x = black_box(i << (n * 8));
                        black_box(x.log10());
                    }
                }
            });
        }

        #[bench]
        fn $random(bench: &mut Bencher) {
            let mut rng = rand::thread_rng();
            /* Exponentially distributed random numbers from the whole range of the type.  */
            let numbers: Vec<$t> = (0..256)
                .map(|_| {
                    let x = rng.gen::<$t>() >> rng.gen_range(0, <$t>::BITS);
                    if x != 0 { x } else { 1 }
                })
                .collect();
            bench.iter(|| {
                for x in &numbers {
                    black_box(black_box(x).log10());
                }
            });
        }

        #[bench]
        fn $random_small(bench: &mut Bencher) {
            let mut rng = rand::thread_rng();
            /* Exponentially distributed random numbers from the range 0..256.  */
            let numbers: Vec<$t> = (0..256)
                .map(|_| {
                    let x = (rng.gen::<u8>() >> rng.gen_range(0, u8::BITS)) as $t;
                    if x != 0 { x } else { 1 }
                })
                .collect();
            bench.iter(|| {
                for x in &numbers {
                    black_box(black_box(x).log10());
                }
            });
        }
    };
}

int_log_bench! {u8, u8_log10_predictable, u8_log10_random, u8_log10_random_small}
int_log_bench! {u16, u16_log10_predictable, u16_log10_random, u16_log10_random_small}
int_log_bench! {u32, u32_log10_predictable, u32_log10_random, u32_log10_random_small}
int_log_bench! {u64, u64_log10_predictable, u64_log10_random, u64_log10_random_small}
int_log_bench! {u128, u128_log10_predictable, u128_log10_random, u128_log10_random_small}
