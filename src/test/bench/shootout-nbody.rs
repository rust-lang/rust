// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// based on:
// http://shootout.alioth.debian.org/u32/benchmark.php?test=nbody&lang=java

extern mod std;

use core::os;

// Using sqrt from the standard library is way slower than using libc
// directly even though std just calls libc, I guess it must be
// because the the indirection through another dynamic linker
// stub. Kind of shocking. Might be able to make it faster still with
// an llvm intrinsic.
#[nolink]
extern mod libc {
    #[legacy_exports];
    fn sqrt(n: float) -> float;
}

fn main() {
    let args = os::args();
    let args = if os::getenv(~"RUST_BENCH").is_some() {
        ~[~"", ~"4000000"]
    } else if args.len() <= 1u {
        ~[~"", ~"100000"]
    } else {
        args
    };
    let n = int::from_str(args[1]).get();
    let mut bodies: ~[Body::props] = NBodySystem::make();
    io::println(fmt!("%f", NBodySystem::energy(bodies)));
    let mut i = 0;
    while i < n {
        NBodySystem::advance(bodies, 0.01);
        i += 1;
    }
    io::println(fmt!("%f", NBodySystem::energy(bodies)));
}

mod NBodySystem {
    use Body;

    pub fn make() -> ~[Body::props] {
        let mut bodies: ~[Body::props] =
            ~[Body::sun(),
              Body::jupiter(),
              Body::saturn(),
              Body::uranus(),
              Body::neptune()];

        let mut px = 0.0;
        let mut py = 0.0;
        let mut pz = 0.0;

        let mut i = 0;
        while i < 5 {
            px += bodies[i].vx * bodies[i].mass;
            py += bodies[i].vy * bodies[i].mass;
            pz += bodies[i].vz * bodies[i].mass;

            i += 1;
        }

        // side-effecting
        Body::offset_momentum(&mut bodies[0], px, py, pz);

        return bodies;
    }

    pub fn advance(bodies: &mut [Body::props], dt: float) {
        let mut i = 0;
        while i < 5 {
            let mut j = i + 1;
            while j < 5 {
                advance_one(&mut bodies[i],
                            &mut bodies[j], dt);
                j += 1;
            }

            i += 1;
        }

        i = 0;
        while i < 5 {
            move_(&mut bodies[i], dt);
            i += 1;
        }
    }

    pub fn advance_one(bi: &mut Body::props,
                       bj: &mut Body::props,
                       dt: float) unsafe {
        let dx = bi.x - bj.x;
        let dy = bi.y - bj.y;
        let dz = bi.z - bj.z;

        let dSquared = dx * dx + dy * dy + dz * dz;

        let distance = ::libc::sqrt(dSquared);
        let mag = dt / (dSquared * distance);

        bi.vx -= dx * bj.mass * mag;
        bi.vy -= dy * bj.mass * mag;
        bi.vz -= dz * bj.mass * mag;

        bj.vx += dx * bi.mass * mag;
        bj.vy += dy * bi.mass * mag;
        bj.vz += dz * bi.mass * mag;
    }

    pub fn move_(b: &mut Body::props, dt: float) {
        b.x += dt * b.vx;
        b.y += dt * b.vy;
        b.z += dt * b.vz;
    }

    pub fn energy(bodies: &[Body::props]) -> float unsafe {
        let mut dx;
        let mut dy;
        let mut dz;
        let mut distance;
        let mut e = 0.0;

        let mut i = 0;
        while i < 5 {
            e +=
                0.5 * bodies[i].mass *
                (bodies[i].vx * bodies[i].vx + bodies[i].vy * bodies[i].vy
                 + bodies[i].vz * bodies[i].vz);

            let mut j = i + 1;
            while j < 5 {
                dx = bodies[i].x - bodies[j].x;
                dy = bodies[i].y - bodies[j].y;
                dz = bodies[i].z - bodies[j].z;

                distance = ::libc::sqrt(dx * dx + dy * dy + dz * dz);
                e -= bodies[i].mass * bodies[j].mass / distance;

                j += 1;
            }

            i += 1;
        }
        return e;

    }
}

mod Body {
    use Body;

    pub const PI: float = 3.141592653589793;
    pub const SOLAR_MASS: float = 39.478417604357432;
    // was 4 * PI * PI originally
    pub const DAYS_PER_YEAR: float = 365.24;

    pub type props =
        {mut x: float,
         mut y: float,
         mut z: float,
         mut vx: float,
         mut vy: float,
         mut vz: float,
         mass: float};

    pub fn jupiter() -> Body::props {
        return {mut x: 4.84143144246472090e+00,
             mut y: -1.16032004402742839e+00,
             mut z: -1.03622044471123109e-01,
             mut vx: 1.66007664274403694e-03 * DAYS_PER_YEAR,
             mut vy: 7.69901118419740425e-03 * DAYS_PER_YEAR,
             mut vz: -6.90460016972063023e-05 * DAYS_PER_YEAR,
             mass: 9.54791938424326609e-04 * SOLAR_MASS};
    }

    pub fn saturn() -> Body::props {
        return {mut x: 8.34336671824457987e+00,
             mut y: 4.12479856412430479e+00,
             mut z: -4.03523417114321381e-01,
             mut vx: -2.76742510726862411e-03 * DAYS_PER_YEAR,
             mut vy: 4.99852801234917238e-03 * DAYS_PER_YEAR,
             mut vz: 2.30417297573763929e-05 * DAYS_PER_YEAR,
             mass: 2.85885980666130812e-04 * SOLAR_MASS};
    }

    pub fn uranus() -> Body::props {
        return {mut x: 1.28943695621391310e+01,
             mut y: -1.51111514016986312e+01,
             mut z: -2.23307578892655734e-01,
             mut vx: 2.96460137564761618e-03 * DAYS_PER_YEAR,
             mut vy: 2.37847173959480950e-03 * DAYS_PER_YEAR,
             mut vz: -2.96589568540237556e-05 * DAYS_PER_YEAR,
             mass: 4.36624404335156298e-05 * SOLAR_MASS};
    }

    pub fn neptune() -> Body::props {
        return {mut x: 1.53796971148509165e+01,
             mut y: -2.59193146099879641e+01,
             mut z: 1.79258772950371181e-01,
             mut vx: 2.68067772490389322e-03 * DAYS_PER_YEAR,
             mut vy: 1.62824170038242295e-03 * DAYS_PER_YEAR,
             mut vz: -9.51592254519715870e-05 * DAYS_PER_YEAR,
             mass: 5.15138902046611451e-05 * SOLAR_MASS};
    }

    pub fn sun() -> Body::props {
        return {mut x: 0.0,
             mut y: 0.0,
             mut z: 0.0,
             mut vx: 0.0,
             mut vy: 0.0,
             mut vz: 0.0,
             mass: SOLAR_MASS};
    }

    pub fn offset_momentum(props: &mut Body::props,
                       px: float, py: float, pz: float) {
        props.vx = -px / SOLAR_MASS;
        props.vy = -py / SOLAR_MASS;
        props.vz = -pz / SOLAR_MASS;
    }

}
