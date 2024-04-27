#![feature(portable_simd)]

use core_simd::simd::prelude::*;

fn a(i: usize, j: usize) -> f64 {
    ((i + j) * (i + j + 1) / 2 + i + 1) as f64
}

fn mult_av(v: &[f64], out: &mut [f64]) {
    assert!(v.len() == out.len());
    assert!(v.len() % 2 == 0);

    for (i, out) in out.iter_mut().enumerate() {
        let mut sum = f64x2::splat(0.0);

        let mut j = 0;
        while j < v.len() {
            let b = f64x2::from_slice(&v[j..]);
            let a = f64x2::from_array([a(i, j), a(i, j + 1)]);
            sum += b / a;
            j += 2
        }
        *out = sum.reduce_sum();
    }
}

fn mult_atv(v: &[f64], out: &mut [f64]) {
    assert!(v.len() == out.len());
    assert!(v.len() % 2 == 0);

    for (i, out) in out.iter_mut().enumerate() {
        let mut sum = f64x2::splat(0.0);

        let mut j = 0;
        while j < v.len() {
            let b = f64x2::from_slice(&v[j..]);
            let a = f64x2::from_array([a(j, i), a(j + 1, i)]);
            sum += b / a;
            j += 2
        }
        *out = sum.reduce_sum();
    }
}

fn mult_atav(v: &[f64], out: &mut [f64], tmp: &mut [f64]) {
    mult_av(v, tmp);
    mult_atv(tmp, out);
}

pub fn spectral_norm(n: usize) -> f64 {
    assert!(n % 2 == 0, "only even lengths are accepted");

    let mut u = vec![1.0; n];
    let mut v = u.clone();
    let mut tmp = u.clone();

    for _ in 0..10 {
        mult_atav(&u, &mut v, &mut tmp);
        mult_atav(&v, &mut u, &mut tmp);
    }
    (dot(&u, &v) / dot(&v, &v)).sqrt()
}

fn dot(x: &[f64], y: &[f64]) -> f64 {
    // This is auto-vectorized:
    x.iter().zip(y).map(|(&x, &y)| x * y).sum()
}

#[cfg(test)]
#[test]
fn test() {
    assert_eq!(format!("{:.9}", spectral_norm(100)), "1.274219991");
}

fn main() {
    // Empty main to make cargo happy
}
