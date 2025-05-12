//@ revisions: old generic generic_with_fn
//@[old]run-pass
//@[generic_with_fn]run-pass
#![feature(
    repr_simd,
    core_intrinsics,
    intrinsics,
    adt_const_params,
    unsized_const_params,
    generic_const_exprs
)]
#![allow(incomplete_features)]

#[cfg(old)]
use std::intrinsics::simd::simd_shuffle;

#[cfg(any(generic, generic_with_fn))]
#[rustc_intrinsic]
unsafe fn simd_shuffle_const_generic<T, U, const I: &'static [u32]>(a: T, b: T) -> U;

#[derive(Copy, Clone)]
#[repr(simd)]
struct Simd<T, const N: usize>([T; N]);

trait Shuffle<const N: usize> {
    const I: Simd<u32, N>;
    const J: &'static [u32] = &Self::I.0;

    unsafe fn shuffle<T, const M: usize>(&self, a: Simd<T, M>, b: Simd<T, M>) -> Simd<T, N>
    where
        Thing<{ Self::J }>:,
    {
        #[cfg(old)]
        return simd_shuffle(a, b, Self::I);
        #[cfg(generic)]
        return simd_shuffle_const_generic::<_, _, { &Self::I.0 }>(a, b);
        //[generic]~^ ERROR overly complex generic constant
        #[cfg(generic_with_fn)]
        return simd_shuffle_const_generic::<_, _, { Self::J }>(a, b);
    }
}

struct Thing<const X: &'static [u32]>;

fn main() {
    struct I1;
    impl Shuffle<4> for I1 {
        const I: Simd<u32, 4> = Simd([0, 2, 4, 6]);
    }

    struct I2;
    impl Shuffle<2> for I2 {
        const I: Simd<u32, 2> = Simd([1, 5]);
    }

    let a = Simd::<u8, 4>([0, 1, 2, 3]);
    let b = Simd::<u8, 4>([4, 5, 6, 7]);
    unsafe {
        let x: Simd<u8, 4> = I1.shuffle(a, b);
        assert_eq!(x.0, [0, 2, 4, 6]);

        let y: Simd<u8, 2> = I2.shuffle(a, b);
        assert_eq!(y.0, [1, 5]);
    }
}
