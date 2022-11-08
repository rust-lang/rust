#![feature(test)]
extern crate test;

use rand::Rng;
use test::Bencher;

macro_rules! unary {
  ($($func:ident),*) => ($(
      paste::item! {
        #[bench]
        pub fn [<$func>](bh: &mut Bencher) {
            let mut rng = rand::thread_rng();
            let x = rng.gen::<f64>();
            bh.iter(|| test::black_box(libm::[<$func>](x)))
        }
        #[bench]
        pub fn [<$func f>](bh: &mut Bencher) {
            let mut rng = rand::thread_rng();
            let x = rng.gen::<f32>();
            bh.iter(|| test::black_box(libm::[<$func f>](x)))
        }
    }
  )*);
}
macro_rules! binary {
  ($($func:ident),*) => ($(
      paste::item! {
        #[bench]
        pub fn [<$func>](bh: &mut Bencher) {
            let mut rng = rand::thread_rng();
            let x = rng.gen::<f64>();
            let y = rng.gen::<f64>();
            bh.iter(|| test::black_box(libm::[<$func>](x, y)))
        }
        #[bench]
        pub fn [<$func f>](bh: &mut Bencher) {
            let mut rng = rand::thread_rng();
            let x = rng.gen::<f32>();
            let y = rng.gen::<f32>();
            bh.iter(|| test::black_box(libm::[<$func f>](x, y)))
        }
    }
  )*);
  ($($func:ident);*) => ($(
      paste::item! {
        #[bench]
        pub fn [<$func>](bh: &mut Bencher) {
            let mut rng = rand::thread_rng();
            let x = rng.gen::<f64>();
            let n = rng.gen::<i32>();
            bh.iter(|| test::black_box(libm::[<$func>](x, n)))
        }
        #[bench]
        pub fn [<$func f>](bh: &mut Bencher) {
            let mut rng = rand::thread_rng();
            let x = rng.gen::<f32>();
            let n = rng.gen::<i32>();
            bh.iter(|| test::black_box(libm::[<$func f>](x, n)))
        }
    }
  )*);
}
macro_rules! trinary {
  ($($func:ident),*) => ($(
      paste::item! {
        #[bench]
        pub fn [<$func>](bh: &mut Bencher) {
            let mut rng = rand::thread_rng();
            let x = rng.gen::<f64>();
            let y = rng.gen::<f64>();
            let z = rng.gen::<f64>();
            bh.iter(|| test::black_box(libm::[<$func>](x, y, z)))
        }
        #[bench]
        pub fn [<$func f>](bh: &mut Bencher) {
            let mut rng = rand::thread_rng();
            let x = rng.gen::<f32>();
            let y = rng.gen::<f32>();
            let z = rng.gen::<f32>();
            bh.iter(|| test::black_box(libm::[<$func f>](x, y, z)))
        }
    }
  )*);
}
macro_rules! bessel {
  ($($func:ident),*) => ($(
      paste::item! {
        #[bench]
        pub fn [<$func>](bh: &mut Bencher) {
            let mut rng = rand::thread_rng();
            let mut n = rng.gen::<i32>();
            n &= 0xffff;
            let x = rng.gen::<f64>();
            bh.iter(|| test::black_box(libm::[<$func>](n, x)))
        }
        #[bench]
        pub fn [<$func f>](bh: &mut Bencher) {
            let mut rng = rand::thread_rng();
            let mut n = rng.gen::<i32>();
            n &= 0xffff;
            let x = rng.gen::<f32>();
            bh.iter(|| test::black_box(libm::[<$func f>](n, x)))
        }
    }
  )*);
}

unary!(
    acos, acosh, asin, atan, cbrt, ceil, cos, cosh, erf, exp, exp2, exp10, expm1, fabs, floor, j0,
    j1, lgamma, log, log1p, log2, log10, rint, round, sin, sinh, sqrt, tan, tanh, tgamma, trunc,
    y0, y1
);
binary!(atan2, copysign, fdim, fmax, fmin, fmod, hypot, pow);
trinary!(fma);
bessel!(jn, yn);
binary!(ldexp; scalbn);
