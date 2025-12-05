const BYTES: usize = 1 << 10;

macro_rules! bench_template {
    ($op:path, $name:ident, $mask:expr) => {
        #[bench]
        fn $name(bench: &mut ::test::Bencher) {
            use ::rand::Rng;
            let mut rng = crate::bench_rng();
            let mut dst = vec![0; ITERATIONS];
            let src1: Vec<U> = (0..ITERATIONS).map(|_| rng.random_range(0..=U::MAX)).collect();
            let mut src2: Vec<U> = (0..ITERATIONS).map(|_| rng.random_range(0..=U::MAX)).collect();
            // Fix the loop invariant mask
            src2[0] = U::MAX / 3;
            let dst = dst.first_chunk_mut().unwrap();
            let src1 = src1.first_chunk().unwrap();
            let src2 = src2.first_chunk().unwrap();

            #[allow(unused)]
            fn vectored(dst: &mut Data, src1: &Data, src2: &Data) {
                let mask = $mask;
                for k in 0..ITERATIONS {
                    dst[k] = $op(src1[k], mask(src2, k));
                }
            }
            let f: fn(&mut Data, &Data, &Data) = vectored;
            let f = ::test::black_box(f);

            bench.iter(|| {
                f(dst, src1, src2);
            });
        }
    };
}

macro_rules! bench_type {
    ($U:ident) => {
        mod $U {
            type U = $U;
            const ITERATIONS: usize = super::BYTES / size_of::<U>();
            type Data = [U; ITERATIONS];
            bench_mask_kind!(constant, |_, _| const { U::MAX / 3 });
            bench_mask_kind!(invariant, |src: &Data, _| src[0]);
            bench_mask_kind!(variable, |src: &Data, k| src[k]);
        }
    };
}

macro_rules! bench_mask_kind {
    ($mask_kind:ident, $mask:expr) => {
        mod $mask_kind {
            use super::{Data, ITERATIONS, U};
            bench_template!(U::gather_bits, gather_bits, $mask);
            bench_template!(U::scatter_bits, scatter_bits, $mask);
        }
    };
}

bench_type!(u8);
bench_type!(u16);
bench_type!(u32);
bench_type!(u64);
bench_type!(u128);
