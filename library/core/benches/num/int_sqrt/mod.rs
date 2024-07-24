//! This benchmarks the `Integer::isqrt` methods.

macro_rules! benches {
    ($($T:ident)+) => {
        $(
            mod $T {
                use test::{black_box, Bencher};

                // Benchmark the square roots of:
                //
                // * the first 1,024 perfect squares
                // * halfway between each of the first 1,024 perfect squares
                //   and the next perfect square
                // * the next perfect square after the each of the first 1,024
                //   perfect squares, minus one
                // * the last 1,024 perfect squares
                // * the last 1,024 perfect squares, minus one
                // * halfway between each of the last 1,024 perfect squares
                //   and the previous perfect square
                #[bench]
                fn isqrt(bench: &mut Bencher) {
                    let mut inputs = Vec::with_capacity(6 * 1_024);

                    // The inputs to benchmark are worked out by using the fact
                    // that the nth nonzero perfect square is the sum of the
                    // first n odd numbers:
                    //
                    //  1 = 1
                    //  4 = 1 + 3
                    //  9 = 1 + 3 + 5
                    // 16 = 1 + 3 + 5 + 7
                    //
                    // Note also that the last odd number added in is two times
                    // the square root of the previous perfect square, plus
                    // one:
                    //
                    // 1 = 2*0 + 1
                    // 3 = 2*1 + 1
                    // 5 = 2*2 + 1
                    // 7 = 2*3 + 1
                    //
                    // That means we can add the square root of this perfect
                    // square once to get about halfway to the next perfect
                    // square, then we can add the square root of this perfect
                    // square again to get to the next perfect square minus
                    // one, then we can add one to get to the next perfect
                    // square.
                    //
                    // Here we include, for each of the first 1,024 perfect
                    // squares:
                    //
                    // * the current perfect square
                    // * about halfway to the next perfect square
                    // * the next perfect square, minus one
                    let mut n: $T = 0;
                    for sqrt_n in 0..1_024.min((1_u128 << (($T::BITS - $T::MAX.leading_zeros())/2)) - 1) as $T {
                        inputs.push(n);
                        n += sqrt_n;
                        inputs.push(n);
                        n += sqrt_n;
                        inputs.push(n);
                        n += 1;
                    }

                    // Similarly, we include, for each of the last 1,024
                    // perfect squares:
                    //
                    // * the current perfect square
                    // * the current perfect square, minus one
                    // * about halfway to the previous perfect square
                    let maximum_sqrt = $T::MAX.isqrt();
                    let mut n = maximum_sqrt * maximum_sqrt;

                    for sqrt_n in (maximum_sqrt - 1_024.min((1_u128 << (($T::BITS - 1)/2)) - 1) as $T..maximum_sqrt).rev() {
                        inputs.push(n);
                        n -= 1;
                        inputs.push(n);
                        n -= sqrt_n;
                        inputs.push(n);
                        n -= sqrt_n;
                    }

                    bench.iter(|| {
                        for x in &inputs {
                            black_box(black_box(x).isqrt());
                        }
                    });
                }
            }
        )*
    };
}

macro_rules! push_n {
    ($T:ident, $inputs:ident, $n:ident) => {
        if $n != 0 {
            $inputs.push(
                core::num::$T::new($n)
                    .expect("Cannot create a new `NonZero` value from a nonzero value"),
            );
        }
    };
}

macro_rules! nonzero_benches {
    ($mod:ident $T:ident $RegularT:ident) => {
        mod $mod {
            use test::{black_box, Bencher};

            // Benchmark the square roots of:
            //
            // * the first 1,024 perfect squares
            // * halfway between each of the first 1,024 perfect squares
            //   and the next perfect square
            // * the next perfect square after the each of the first 1,024
            //   perfect squares, minus one
            // * the last 1,024 perfect squares
            // * the last 1,024 perfect squares, minus one
            // * halfway between each of the last 1,024 perfect squares
            //   and the previous perfect square
            #[bench]
            fn isqrt(bench: &mut Bencher) {
                let mut inputs: Vec<core::num::$T> = Vec::with_capacity(6 * 1_024);

                // The inputs to benchmark are worked out by using the fact
                // that the nth nonzero perfect square is the sum of the
                // first n odd numbers:
                //
                //  1 = 1
                //  4 = 1 + 3
                //  9 = 1 + 3 + 5
                // 16 = 1 + 3 + 5 + 7
                //
                // Note also that the last odd number added in is two times
                // the square root of the previous perfect square, plus
                // one:
                //
                // 1 = 2*0 + 1
                // 3 = 2*1 + 1
                // 5 = 2*2 + 1
                // 7 = 2*3 + 1
                //
                // That means we can add the square root of this perfect
                // square once to get about halfway to the next perfect
                // square, then we can add the square root of this perfect
                // square again to get to the next perfect square minus
                // one, then we can add one to get to the next perfect
                // square.
                //
                // Here we include, for each of the first 1,024 perfect
                // squares:
                //
                // * the current perfect square
                // * about halfway to the next perfect square
                // * the next perfect square, minus one
                let mut n: $RegularT = 0;
                for sqrt_n in 0..1_024
                    .min((1_u128 << (($RegularT::BITS - $RegularT::MAX.leading_zeros()) / 2)) - 1)
                    as $RegularT
                {
                    push_n!($T, inputs, n);
                    n += sqrt_n;
                    push_n!($T, inputs, n);
                    n += sqrt_n;
                    push_n!($T, inputs, n);
                    n += 1;
                }

                // Similarly, we include, for each of the last 1,024
                // perfect squares:
                //
                // * the current perfect square
                // * the current perfect square, minus one
                // * about halfway to the previous perfect square
                let maximum_sqrt = $RegularT::MAX.isqrt();
                let mut n = maximum_sqrt * maximum_sqrt;

                for sqrt_n in (maximum_sqrt
                    - 1_024.min((1_u128 << (($RegularT::BITS - 1) / 2)) - 1) as $RegularT
                    ..maximum_sqrt)
                    .rev()
                {
                    push_n!($T, inputs, n);
                    n -= 1;
                    push_n!($T, inputs, n);
                    n -= sqrt_n;
                    push_n!($T, inputs, n);
                    n -= sqrt_n;
                }

                bench.iter(|| {
                    for n in &inputs {
                        black_box(black_box(n).isqrt());
                    }
                });
            }
        }
    };
}

benches!(i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize);
nonzero_benches!(non_zero_u8 NonZeroU8 u8);
nonzero_benches!(non_zero_u16 NonZeroU16 u16);
nonzero_benches!(non_zero_u32 NonZeroU32 u32);
nonzero_benches!(non_zero_u64 NonZeroU64 u64);
nonzero_benches!(non_zero_u128 NonZeroU128 u128);
nonzero_benches!(non_zero_usize NonZeroUsize usize);
