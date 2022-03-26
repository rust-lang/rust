// Code taken from the `packed_simd` crate
// Run this code with `cargo test --example dot_product`
#![feature(array_chunks)]
use core_simd::*;

/// This is your barebones dot product implementation: 
/// Take 2 vectors, multiply them element wise and *then*
/// add up the result. In the next example we will see if there
///  is any difference to adding as we go along multiplying.
pub fn dot_prod_0(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    a.iter()
    .zip(b.iter())
    .map(|a, b| a * b)
    .sum()
}

pub fn dot_prod_1(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
    .zip(b.iter())
    .fold(0.0, |a, b| a * b)
}

pub fn dot_prod_simd_0(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    // TODO handle remainder when a.len() % 4 != 0
    a.array_chunks::<4>()
        .map(|&a| f32x4::from_array(a))
        .zip(b.array_chunks::<4>().map(|&b| f32x4::from_array(b)))
        .map(|(a, b)| (a * b).horizontal_sum())
        .sum()
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

        assert_eq!(0.0, dot_prod_0(&a, &b));
        assert_eq!(0.0, dot_prod_1(&a, &b));
        assert_eq!(0.0, dot_prod_simd_0(&a, &b));
        assert_eq!(0.0, dot_prod_simd_1(&a, &b));
    }
}
