//! n-body benchmark from the [benchmarks game][bg].
//!
//! [bg]: https://benchmarksgame.alioth.debian.org/u64q/nbody-description.
//! html#nbody

#![cfg_attr(stdsimd_strict, deny(warnings))]
#![feature(stdsimd)]
#![cfg_attr(feature = "cargo-clippy",
            allow(similar_names, missing_docs_in_private_items,
                  shadow_reuse, print_stdout))]

extern crate stdsimd;
use stdsimd::simd::*;

const PI: f64 = std::f64::consts::PI;
const SOLAR_MASS: f64 = 4.0 * PI * PI;
const DAYS_PER_YEAR: f64 = 365.24;

pub trait Frsqrt {
    fn frsqrt(&self) -> Self;
}

impl Frsqrt for f64x2 {
    fn frsqrt(&self) -> Self {
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"),
                  target_feature = "sse"))]
        {
            #[cfg(target_arch = "x86")]
            use stdsimd::arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use stdsimd::arch::x86_64::*;
            let t: f32x2 = (*self).into();

            let u: f64x4 = unsafe {
                let res = _mm_rsqrt_ps(_mm_setr_ps(
                    t.extract(0),
                    t.extract(1),
                    0.,
                    0.,
                ));
                f32x4::from_bits(res).into()
            };
            Self::new(u.extract(0), u.extract(1))
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            #[cfg(target_arch = "aarch64")]
            use stdsimd::arch::aarch64::*;
            #[cfg(target_arch = "arm")]
            use stdsimd::arch::arm::*;

            let t: f32x2 = (*self).into();
            let t: f32x2 = unsafe { vrsqrte_f32(t.into_bits()).into_bits() };
            t.into()
        }
        #[cfg(not(any(all(any(target_arch = "x86",
                              target_arch = "x86_64"),
                          target_feature = "sse"),
                      all(target_arch = "aarch64",
                          target_feature = "neon"))))]
        {
            self.replace(0, 1. / self.extract(0).sqrt());
            self.replace(1, 1. / self.extract(1).sqrt());
            *self
        }
    }
}

struct Body {
    x: [f64; 3],
    _fill: f64,
    v: [f64; 3],
    mass: f64,
}

impl Body {
    fn new(
        x0: f64, x1: f64, x2: f64, v0: f64, v1: f64, v2: f64, mass: f64
    ) -> Self {
        Self {
            x: [x0, x1, x2],
            _fill: 0.0,
            v: [v0, v1, v2],
            mass,
        }
    }
}

const N_BODIES: usize = 5;
const N: usize = N_BODIES * (N_BODIES - 1) / 2;
fn offset_momentum(bodies: &mut [Body; N_BODIES]) {
    let (sun, rest) = bodies.split_at_mut(1);
    let sun = &mut sun[0];
    for body in rest {
        for k in 0..3 {
            sun.v[k] -= body.v[k] * body.mass / SOLAR_MASS;
        }
    }
}
fn advance(bodies: &mut [Body; N_BODIES], dt: f64) {
    let mut r = [[0.0; 4]; N];
    let mut mag = [0.0; N];

    let mut dx = [f64x2::splat(0.0); 3];
    let mut dsquared;
    let mut distance;
    let mut dmag;

    let mut i = 0;
    for j in 0..N_BODIES {
        for k in j + 1..N_BODIES {
            for m in 0..3 {
                r[i][m] = bodies[j].x[m] - bodies[k].x[m];
            }
            i += 1;
        }
    }

    i = 0;
    while i < N {
        for (m, dx) in dx.iter_mut().enumerate() {
            *dx = f64x2::new(r[i][m], r[i + 1][m]);
        }

        dsquared = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
        distance = dsquared.frsqrt();
        for _ in 0..2 {
            distance = distance * f64x2::splat(1.5)
                - ((f64x2::splat(0.5) * dsquared) * distance)
                    * (distance * distance)
        }
        dmag = f64x2::splat(dt) / dsquared * distance;
        dmag.store_unaligned(&mut mag[i..]);

        i += 2;
    }

    i = 0;
    for j in 0..N_BODIES {
        for k in j + 1..N_BODIES {
            for m in 0..3 {
                bodies[j].v[m] -= r[i][m] * bodies[k].mass * mag[i];
                bodies[k].v[m] += r[i][m] * bodies[j].mass * mag[i];
            }
            i += 1
        }
    }
    for body in bodies {
        for m in 0..3 {
            body.x[m] += dt * body.v[m]
        }
    }
}

fn energy(bodies: &[Body; N_BODIES]) -> f64 {
    let mut e = 0.0;
    for i in 0..N_BODIES {
        let bi = &bodies[i];
        e += bi.mass
            * (bi.v[0] * bi.v[0] + bi.v[1] * bi.v[1] + bi.v[2] * bi.v[2])
            / 2.0;
        for bj in bodies.iter().take(N_BODIES).skip(i + 1) {
            let mut dx = [0.0; 3];
            for (k, dx) in dx.iter_mut().enumerate() {
                *dx = bi.x[k] - bj.x[k];
            }
            let mut distance = 0.0;
            for &d in &dx {
                distance += d * d
            }
            e -= bi.mass * bj.mass / distance.sqrt()
        }
    }
    e
}

fn main() {
    let mut bodies: [Body; N_BODIES] = [
        /* sun */
        Body::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, SOLAR_MASS),
        /* jupiter */
        Body::new(
            4.84143144246472090e+00,
            -1.16032004402742839e+00,
            -1.03622044471123109e-01,
            1.66007664274403694e-03 * DAYS_PER_YEAR,
            7.69901118419740425e-03 * DAYS_PER_YEAR,
            -6.90460016972063023e-05 * DAYS_PER_YEAR,
            9.54791938424326609e-04 * SOLAR_MASS,
        ),
        /* saturn */
        Body::new(
            8.34336671824457987e+00,
            4.12479856412430479e+00,
            -4.03523417114321381e-01,
            -2.76742510726862411e-03 * DAYS_PER_YEAR,
            4.99852801234917238e-03 * DAYS_PER_YEAR,
            2.30417297573763929e-05 * DAYS_PER_YEAR,
            2.85885980666130812e-04 * SOLAR_MASS,
        ),
        /* uranus */
        Body::new(
            1.28943695621391310e+01,
            -1.51111514016986312e+01,
            -2.23307578892655734e-01,
            2.96460137564761618e-03 * DAYS_PER_YEAR,
            2.37847173959480950e-03 * DAYS_PER_YEAR,
            -2.96589568540237556e-05 * DAYS_PER_YEAR,
            4.36624404335156298e-05 * SOLAR_MASS,
        ),
        /* neptune */
        Body::new(
            1.53796971148509165e+01,
            -2.59193146099879641e+01,
            1.79258772950371181e-01,
            2.68067772490389322e-03 * DAYS_PER_YEAR,
            1.62824170038242295e-03 * DAYS_PER_YEAR,
            -9.51592254519715870e-05 * DAYS_PER_YEAR,
            5.15138902046611451e-05 * SOLAR_MASS,
        ),
    ];

    let n: usize = std::env::args()
        .nth(1)
        .expect("need one arg")
        .parse()
        .expect("argument should be a usize");

    offset_momentum(&mut bodies);
    println!("{:.9}", energy(&bodies));
    for _ in 0..n {
        advance(&mut bodies, 0.01);
    }
    println!("{:.9}", energy(&bodies));
}
