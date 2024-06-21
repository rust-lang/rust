//! Code taken from the `packed_simd` crate.
//! Run this code with `cargo test --example dot_product`.

#![feature(array_chunks)]
#![feature(slice_as_chunks)]
// Add these imports to use the stdsimd library
#![feature(portable_simd)]
use core_simd::simd::prelude::*;

// This is your barebones dot product implementation:
// Take 2 vectors, multiply them element wise and *then*
// go along the resulting array and add up the result.
// In the next example we will see if there
//  is any difference to adding and multiplying in tandem.
pub fn dot_prod_scalar_0(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    a.iter().zip(b.iter()).map(|(a, b)| a * b).sum()
}

// When dealing with SIMD, it is very important to think about the amount
// of data movement and when it happens. We're going over simple computation examples here, and yet
// it is not trivial to understand what may or may not contribute to performance
// changes. Eventually, you will need tools to inspect the generated assembly and confirm your
// hypothesis and benchmarks - we will mention them later on.
// With the use of `fold`, we're doing a multiplication,
// and then adding it to the sum, one element from both vectors at a time.
pub fn dot_prod_scalar_1(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .fold(0.0, |a, zipped| a + zipped.0 * zipped.1)
}

// We now move on to the SIMD implementations: notice the following constructs:
// `array_chunks::<4>`: mapping this over the vector will let use construct SIMD vectors
// `f32x4::from_array`: construct the SIMD vector from a slice
// `(a * b).reduce_sum()`: Multiply both f32x4 vectors together, and then reduce them.
// This approach essentially uses SIMD to produce a vector of length N/4 of all the products,
// and then add those with `sum()`. This is suboptimal.
// TODO: ASCII diagrams
pub fn dot_prod_simd_0(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    // TODO handle remainder when a.len() % 4 != 0
    a.array_chunks::<4>()
        .map(|&a| f32x4::from_array(a))
        .zip(b.array_chunks::<4>().map(|&b| f32x4::from_array(b)))
        .map(|(a, b)| (a * b).reduce_sum())
        .sum()
}

// There's some simple ways to improve the previous code:
// 1. Make a `zero` `f32x4` SIMD vector that we will be accumulating into
// So that there is only one `sum()` reduction when the last `f32x4` has been processed
// 2. Exploit Fused Multiply Add so that the multiplication, addition and sinking into the reduciton
// happen in the same step.
// If the arrays are large, minimizing the data shuffling will lead to great perf.
// If the arrays are small, handling the remainder elements when the length isn't a multiple of 4
// Can become a problem.
pub fn dot_prod_simd_1(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    // TODO handle remainder when a.len() % 4 != 0
    a.array_chunks::<4>()
        .map(|&a| f32x4::from_array(a))
        .zip(b.array_chunks::<4>().map(|&b| f32x4::from_array(b)))
        .fold(f32x4::splat(0.0), |acc, zipped| acc + zipped.0 * zipped.1)
        .reduce_sum()
}

// A lot of knowledgeable use of SIMD comes from knowing specific instructions that are
// available - let's try to use the `mul_add` instruction, which is the fused-multiply-add we were looking for.
use std_float::StdFloat;
pub fn dot_prod_simd_2(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    // TODO handle remainder when a.len() % 4 != 0
    let mut res = f32x4::splat(0.0);
    a.array_chunks::<4>()
        .map(|&a| f32x4::from_array(a))
        .zip(b.array_chunks::<4>().map(|&b| f32x4::from_array(b)))
        .for_each(|(a, b)| {
            res = a.mul_add(b, res);
        });
    res.reduce_sum()
}

// Finally, we will write the same operation but handling the loop remainder.
const LANES: usize = 4;
pub fn dot_prod_simd_3(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    let (a_extra, a_chunks) = a.as_rchunks();
    let (b_extra, b_chunks) = b.as_rchunks();

    // These are always true, but for emphasis:
    assert_eq!(a_chunks.len(), b_chunks.len());
    assert_eq!(a_extra.len(), b_extra.len());

    let mut sums = [0.0; LANES];
    for ((x, y), d) in std::iter::zip(a_extra, b_extra).zip(&mut sums) {
        *d = x * y;
    }

    let mut sums = f32x4::from_array(sums);
    std::iter::zip(a_chunks, b_chunks).for_each(|(x, y)| {
        sums += f32x4::from_array(*x) * f32x4::from_array(*y);
    });

    sums.reduce_sum()
}

// Finally, we present an iterator version for handling remainders in a scalar fashion at the end of the loop.
// Unfortunately, this is allocating 1 `XMM` register on the order of `~len(a)` - we'll see how we can get around it in the
// next example.
pub fn dot_prod_simd_4(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = a
        .array_chunks::<4>()
        .map(|&a| f32x4::from_array(a))
        .zip(b.array_chunks::<4>().map(|&b| f32x4::from_array(b)))
        .map(|(a, b)| a * b)
        .fold(f32x4::splat(0.0), std::ops::Add::add)
        .reduce_sum();
    let remain = a.len() - (a.len() % 4);
    sum += a[remain..]
        .iter()
        .zip(&b[remain..])
        .map(|(a, b)| a * b)
        .sum::<f32>();
    sum
}

// This version allocates a single `XMM` register for accumulation, and the folds don't allocate on top of that.
// Notice the use of `mul_add`, which can do a multiply and an add operation ber iteration.
pub fn dot_prod_simd_5(a: &[f32], b: &[f32]) -> f32 {
    a.array_chunks::<4>()
        .map(|&a| f32x4::from_array(a))
        .zip(b.array_chunks::<4>().map(|&b| f32x4::from_array(b)))
        .fold(f32x4::splat(0.), |acc, (a, b)| a.mul_add(b, acc))
        .reduce_sum()
}

fn main() {
    // Empty main to make cargo happy
}

#[cfg(test)]
mod tests {
    #[test]
    fn smoke_test() {
        use super::*;
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b: Vec<f32> = vec![-8.0, -7.0, -6.0, -5.0, 4.0, 3.0, 2.0, 1.0];
        let x: Vec<f32> = [0.5; 1003].to_vec();
        let y: Vec<f32> = [2.0; 1003].to_vec();

        // Basic check
        assert_eq!(0.0, dot_prod_scalar_0(&a, &b));
        assert_eq!(0.0, dot_prod_scalar_1(&a, &b));
        assert_eq!(0.0, dot_prod_simd_0(&a, &b));
        assert_eq!(0.0, dot_prod_simd_1(&a, &b));
        assert_eq!(0.0, dot_prod_simd_2(&a, &b));
        assert_eq!(0.0, dot_prod_simd_3(&a, &b));
        assert_eq!(0.0, dot_prod_simd_4(&a, &b));
        assert_eq!(0.0, dot_prod_simd_5(&a, &b));

        // We can handle vectors that are non-multiples of 4
        assert_eq!(1003.0, dot_prod_simd_3(&x, &y));
    }
}
