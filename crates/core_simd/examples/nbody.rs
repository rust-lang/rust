#![feature(platform_intrinsics, repr_simd)]
use core_simd::*;

use std::f64::consts::PI;
const SOLAR_MASS: f64 = 4.0 * PI * PI;
const DAYS_PER_YEAR: f64 = 365.24;

pub struct Body {
    pub x: f64x4,
    pub v: f64x4,
    pub mass: f64,
}
const N_BODIES: usize = 5;
#[allow(clippy::unreadable_literal)]
const BODIES: [Body; N_BODIES] = [
    // sun:
    Body {
        x: f64x4::new(0., 0., 0., 0.),
        v: f64x4::new(0., 0., 0., 0.),
        mass: SOLAR_MASS,
    },
    // jupiter:
    Body {
        x: f64x4::new(
            4.84143144246472090e+00,
            -1.16032004402742839e+00,
            -1.03622044471123109e-01,
            0.,
        ),
        v: f64x4::new(
            1.66007664274403694e-03 * DAYS_PER_YEAR,
            7.69901118419740425e-03 * DAYS_PER_YEAR,
            -6.90460016972063023e-05 * DAYS_PER_YEAR,
            0.,
        ),
        mass: 9.54791938424326609e-04 * SOLAR_MASS,
    },
    // saturn:
    Body {
        x: f64x4::new(
            8.34336671824457987e+00,
            4.12479856412430479e+00,
            -4.03523417114321381e-01,
            0.,
        ),
        v: f64x4::new(
            -2.76742510726862411e-03 * DAYS_PER_YEAR,
            4.99852801234917238e-03 * DAYS_PER_YEAR,
            2.30417297573763929e-05 * DAYS_PER_YEAR,
            0.,
        ),
        mass: 2.85885980666130812e-04 * SOLAR_MASS,
    },
    // uranus:
    Body {
        x: f64x4::new(
            1.28943695621391310e+01,
            -1.51111514016986312e+01,
            -2.23307578892655734e-01,
            0.,
        ),
        v: f64x4::new(
            2.96460137564761618e-03 * DAYS_PER_YEAR,
            2.37847173959480950e-03 * DAYS_PER_YEAR,
            -2.96589568540237556e-05 * DAYS_PER_YEAR,
            0.,
        ),
        mass: 4.36624404335156298e-05 * SOLAR_MASS,
    },
    // neptune:
    Body {
        x: f64x4::new(
            1.53796971148509165e+01,
            -2.59193146099879641e+01,
            1.79258772950371181e-01,
            0.,
        ),
        v: f64x4::new(
            2.68067772490389322e-03 * DAYS_PER_YEAR,
            1.62824170038242295e-03 * DAYS_PER_YEAR,
            -9.51592254519715870e-05 * DAYS_PER_YEAR,
            0.,
        ),
        mass: 5.15138902046611451e-05 * SOLAR_MASS,
    },
];

pub fn offset_momentum(bodies: &mut [Body; N_BODIES]) {
    let (sun, rest) = bodies.split_at_mut(1);
    let sun = &mut sun[0];
    for body in rest {
        let m_ratio = body.mass / SOLAR_MASS;
        sun.v -= body.v * m_ratio;
    }
}

pub fn energy(bodies: &[Body; N_BODIES]) -> f64 {
    let mut e = 0.;
    for i in 0..N_BODIES {
        let bi = &bodies[i];
        e += bi.mass * (bi.v * bi.v).sum() * 0.5;
        for bj in bodies.iter().take(N_BODIES).skip(i + 1) {
            let dx = bi.x - bj.x;
            e -= bi.mass * bj.mass / (dx * dx).sum().sqrt()
        }
    }
    e
}

pub fn advance(bodies: &mut [Body; N_BODIES], dt: f64) {
    const N: usize = N_BODIES * (N_BODIES - 1) / 2;

    // compute distance between bodies:
    let mut r = [f64x4::splat(0.); N];
    {
        let mut i = 0;
        for j in 0..N_BODIES {
            for k in j + 1..N_BODIES {
                r[i] = bodies[j].x - bodies[k].x;
                i += 1;
            }
        }
    }

    let mut mag = [0.0; N];
    let mut i = 0;
    while i < N {
        let d2s = f64x2::new((r[i] * r[i]).sum(), (r[i + 1] * r[i + 1]).sum());
        let dmags = f64x2::splat(dt) / (d2s * d2s.sqrte());
        dmags.write_to_slice_unaligned(&mut mag[i..]);
        i += 2;
    }

    i = 0;
    for j in 0..N_BODIES {
        for k in j + 1..N_BODIES {
            let f = r[i] * mag[i];
            bodies[j].v -= f * bodies[k].mass;
            bodies[k].v += f * bodies[j].mass;
            i += 1
        }
    }
    for body in bodies {
        body.x += dt * body.v
    }
}

pub fn run_k<K>(n: usize, k: K) -> (f64, f64)
where
    K: Fn(&mut [Body; N_BODIES], f64),
{
    let mut bodies = BODIES;
    offset_momentum(&mut bodies);
    let energy_before = energy(&bodies);
    for _ in 0..n {
        k(&mut bodies, 0.01);
    }
    let energy_after = energy(&bodies);

    (energy_before, energy_after)
}

pub fn run(n: usize) -> (f64, f64) {
    run_k(n, advance)
}

const OUTPUT: Vec<f64> = vec![-0.169075164, -0.169087605];
#[cfg(test)]
mod tests {
    #[test]
    fn test() {
        let mut out: Vec<u8> = Vec::new();
        run(&mut out, 1000, 0);
        for &(size, a_e, b_e) in crate::RESULTS {
            let (a, b) = super::run(size);
            assert_eq!(format!("{:.9}", a), a_e);
            assert_eq!(format!("{:.9}", b), b_e);
        }
    }
}
fn main() {
    //let n: usize = std::env::args()
    //.nth(1)
    //.expect("need one arg")
    //.parse()
    //.expect("argument should be a usize");
    //run(&mut std::io::stdout(), n, alg);
    println!("{:?}", run_k<10>(10, 10));
}
