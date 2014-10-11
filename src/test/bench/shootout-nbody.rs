// The Computer Language Benchmarks Game
// http://benchmarksgame.alioth.debian.org/
//
// contributed by the Rust Project Developers

// Copyright (c) 2011-2014 The Rust Project Developers
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// - Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
//
// - Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in
//   the documentation and/or other materials provided with the
//   distribution.
//
// - Neither the name of "The Computer Language Benchmarks Game" nor
//   the name of "The Computer Language Shootout Benchmarks" nor the
//   names of its contributors may be used to endorse or promote
//   products derived from this software without specific prior
//   written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
// OF THE POSSIBILITY OF SUCH DAMAGE.

const PI: f64 = 3.141592653589793;
const SOLAR_MASS: f64 = 4.0 * PI * PI;
const YEAR: f64 = 365.24;
const N_BODIES: uint = 5;

static BODIES: [Planet, ..N_BODIES] = [
    // Sun
    Planet {
        x: 0.0, y: 0.0, z: 0.0,
        vx: 0.0, vy: 0.0, vz: 0.0,
        mass: SOLAR_MASS,
    },
    // Jupiter
    Planet {
        x: 4.84143144246472090e+00,
        y: -1.16032004402742839e+00,
        z: -1.03622044471123109e-01,
        vx: 1.66007664274403694e-03 * YEAR,
        vy: 7.69901118419740425e-03 * YEAR,
        vz: -6.90460016972063023e-05 * YEAR,
        mass: 9.54791938424326609e-04 * SOLAR_MASS,
    },
    // Saturn
    Planet {
        x: 8.34336671824457987e+00,
        y: 4.12479856412430479e+00,
        z: -4.03523417114321381e-01,
        vx: -2.76742510726862411e-03 * YEAR,
        vy: 4.99852801234917238e-03 * YEAR,
        vz: 2.30417297573763929e-05 * YEAR,
        mass: 2.85885980666130812e-04 * SOLAR_MASS,
    },
    // Uranus
    Planet {
        x: 1.28943695621391310e+01,
        y: -1.51111514016986312e+01,
        z: -2.23307578892655734e-01,
        vx: 2.96460137564761618e-03 * YEAR,
        vy: 2.37847173959480950e-03 * YEAR,
        vz: -2.96589568540237556e-05 * YEAR,
        mass: 4.36624404335156298e-05 * SOLAR_MASS,
    },
    // Neptune
    Planet {
        x: 1.53796971148509165e+01,
        y: -2.59193146099879641e+01,
        z: 1.79258772950371181e-01,
        vx: 2.68067772490389322e-03 * YEAR,
        vy: 1.62824170038242295e-03 * YEAR,
        vz: -9.51592254519715870e-05 * YEAR,
        mass: 5.15138902046611451e-05 * SOLAR_MASS,
    },
];

struct Planet {
    x: f64, y: f64, z: f64,
    vx: f64, vy: f64, vz: f64,
    mass: f64,
}

fn advance(bodies: &mut [Planet, ..N_BODIES], dt: f64, steps: int) {
    for _ in range(0, steps) {
        let mut b_slice = bodies.as_mut_slice();
        loop {
            let bi = match b_slice.mut_shift_ref() {
                Some(bi) => bi,
                None => break
            };
            for bj in b_slice.iter_mut() {
                let dx = bi.x - bj.x;
                let dy = bi.y - bj.y;
                let dz = bi.z - bj.z;

                let d2 = dx * dx + dy * dy + dz * dz;
                let mag = dt / (d2 * d2.sqrt());

                let massj_mag = bj.mass * mag;
                bi.vx -= dx * massj_mag;
                bi.vy -= dy * massj_mag;
                bi.vz -= dz * massj_mag;

                let massi_mag = bi.mass * mag;
                bj.vx += dx * massi_mag;
                bj.vy += dy * massi_mag;
                bj.vz += dz * massi_mag;
            }
            bi.x += dt * bi.vx;
            bi.y += dt * bi.vy;
            bi.z += dt * bi.vz;
        }
    }
}

fn energy(bodies: &[Planet, ..N_BODIES]) -> f64 {
    let mut e = 0.0;
    let mut bodies = bodies.iter();
    loop {
        let bi = match bodies.next() {
            Some(bi) => bi,
            None => break
        };
        e += (bi.vx * bi.vx + bi.vy * bi.vy + bi.vz * bi.vz) * bi.mass / 2.0;
        for bj in bodies.clone() {
            let dx = bi.x - bj.x;
            let dy = bi.y - bj.y;
            let dz = bi.z - bj.z;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            e -= bi.mass * bj.mass / dist;
        }
    }
    e
}

fn offset_momentum(bodies: &mut [Planet, ..N_BODIES]) {
    let mut px = 0.0;
    let mut py = 0.0;
    let mut pz = 0.0;
    for bi in bodies.iter() {
        px += bi.vx * bi.mass;
        py += bi.vy * bi.mass;
        pz += bi.vz * bi.mass;
    }
    let sun = &mut bodies[0];
    sun.vx = - px / SOLAR_MASS;
    sun.vy = - py / SOLAR_MASS;
    sun.vz = - pz / SOLAR_MASS;
}

fn main() {
    let n = if std::os::getenv("RUST_BENCH").is_some() {
        5000000
    } else {
        std::os::args().as_slice().get(1)
            .and_then(|arg| from_str(arg.as_slice()))
            .unwrap_or(1000)
    };
    let mut bodies = BODIES;

    offset_momentum(&mut bodies);
    println!("{:.9f}", energy(&bodies));

    advance(&mut bodies, 0.01, n);

    println!("{:.9f}", energy(&bodies));
}
