// Code taken from the `packed_simd` crate
// Run this code with `cargo test --example dot_product`
#![feature(array_chunks)]
use core_simd::*;

pub fn dot_prod(a: &[f32], b: &[f32]) -> f32 {
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
    fn test() {
        use super::*;
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b: Vec<f32> = vec![-8.0, -7.0, -6.0, -5.0, 4.0, 3.0, 2.0, 1.0];

        assert_eq!(0.0, dot_prod(&a, &b));
    }
}
