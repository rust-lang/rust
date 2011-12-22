// based on:
// http://shootout.alioth.debian.org/u32/benchmark.php?test=nbody&lang=java

#[abi = "cdecl"]
#[nolink]
native mod llvm {
    fn sqrt(n: float) -> float;
}

fn main() {
    //
    // Leave these commented out to
    // finish in a reasonable time
    // during 'make check' under valgrind
    // 5000000
    // 50000000
    let inputs: [int] = [50000, 500000];

    let bodies: [Body::props] = NBodySystem::MakeNBodySystem();


    for n: int in inputs {
        log_full(core::debug, NBodySystem::energy(bodies));

        let i: int = 0;
        while i < n { NBodySystem::advance(bodies, 0.01); i += 1; }
        log_full(core::debug, NBodySystem::energy(bodies));
    }
}

// Body::props is a record of floats, so
// vec<Body::props> is a vector of records of floats

mod NBodySystem {

    fn MakeNBodySystem() -> [Body::props] {
        // these each return a Body::props
        let bodies: [Body::props] =
            [Body::sun(), Body::jupiter(), Body::saturn(), Body::uranus(),
             Body::neptune()];

        let px: float = 0.0;
        let py: float = 0.0;
        let pz: float = 0.0;

        let i: int = 0;
        while i < 5 {
            px += bodies[i].vx * bodies[i].mass;
            py += bodies[i].vy * bodies[i].mass;
            pz += bodies[i].vz * bodies[i].mass;

            i += 1;
        }

        // side-effecting
        Body::offsetMomentum(bodies[0], px, py, pz);

        ret bodies;
    }

    fn advance(bodies: [Body::props], dt: float) {

        let i: int = 0;
        while i < 5 {
            let j: int = i + 1;
            while j < 5 { advance_one(bodies[i], bodies[j], dt); j += 1; }

            i += 1;
        }

        i = 0;
        while i < 5 { move(bodies[i], dt); i += 1; }
    }

    fn advance_one(bi: Body::props, bj: Body::props, dt: float) unsafe {
        let dx: float = bi.x - bj.x;
        let dy: float = bi.y - bj.y;
        let dz: float = bi.z - bj.z;

        let dSquared: float = dx * dx + dy * dy + dz * dz;

        let distance: float = llvm::sqrt(dSquared);
        let mag: float = dt / (dSquared * distance);

        bi.vx -= dx * bj.mass * mag;
        bi.vy -= dy * bj.mass * mag;
        bi.vz -= dz * bj.mass * mag;

        bj.vx += dx * bi.mass * mag;
        bj.vy += dy * bi.mass * mag;
        bj.vz += dz * bi.mass * mag;
    }

    fn move(b: Body::props, dt: float) {
        b.x += dt * b.vx;
        b.y += dt * b.vy;
        b.z += dt * b.vz;
    }

    fn energy(bodies: [Body::props]) -> float unsafe {
        let dx: float;
        let dy: float;
        let dz: float;
        let distance: float;
        let e: float = 0.0;

        let i: int = 0;
        while i < 5 {
            e +=
                0.5 * bodies[i].mass *
                    (bodies[i].vx * bodies[i].vx + bodies[i].vy * bodies[i].vy
                         + bodies[i].vz * bodies[i].vz);

            let j: int = i + 1;
            while j < 5 {
                dx = bodies[i].x - bodies[j].x;
                dy = bodies[i].y - bodies[j].y;
                dz = bodies[i].z - bodies[j].z;

                distance = llvm::sqrt(dx * dx + dy * dy + dz * dz);
                e -= bodies[i].mass * bodies[j].mass / distance;

                j += 1;
            }

            i += 1;
        }
        ret e;

    }
}

mod Body {

    const PI: float = 3.141592653589793;
    const SOLAR_MASS: float = 39.478417604357432;
    // was 4 * PI * PI originally
    const DAYS_PER_YEAR: float = 365.24;

    type props =
        {mutable x: float,
         mutable y: float,
         mutable z: float,
         mutable vx: float,
         mutable vy: float,
         mutable vz: float,
         mass: float};

    fn jupiter() -> Body::props {
        ret {mutable x: 4.84143144246472090e+00,
             mutable y: -1.16032004402742839e+00,
             mutable z: -1.03622044471123109e-01,
             mutable vx: 1.66007664274403694e-03 * DAYS_PER_YEAR,
             mutable vy: 7.69901118419740425e-03 * DAYS_PER_YEAR,
             mutable vz: -6.90460016972063023e-05 * DAYS_PER_YEAR,
             mass: 9.54791938424326609e-04 * SOLAR_MASS};
    }

    fn saturn() -> Body::props {
        ret {mutable x: 8.34336671824457987e+00,
             mutable y: 4.12479856412430479e+00,
             mutable z: -4.03523417114321381e-01,
             mutable vx: -2.76742510726862411e-03 * DAYS_PER_YEAR,
             mutable vy: 4.99852801234917238e-03 * DAYS_PER_YEAR,
             mutable vz: 2.30417297573763929e-05 * DAYS_PER_YEAR,
             mass: 2.85885980666130812e-04 * SOLAR_MASS};
    }

    fn uranus() -> Body::props {
        ret {mutable x: 1.28943695621391310e+01,
             mutable y: -1.51111514016986312e+01,
             mutable z: -2.23307578892655734e-01,
             mutable vx: 2.96460137564761618e-03 * DAYS_PER_YEAR,
             mutable vy: 2.37847173959480950e-03 * DAYS_PER_YEAR,
             mutable vz: -2.96589568540237556e-05 * DAYS_PER_YEAR,
             mass: 4.36624404335156298e-05 * SOLAR_MASS};
    }

    fn neptune() -> Body::props {
        ret {mutable x: 1.53796971148509165e+01,
             mutable y: -2.59193146099879641e+01,
             mutable z: 1.79258772950371181e-01,
             mutable vx: 2.68067772490389322e-03 * DAYS_PER_YEAR,
             mutable vy: 1.62824170038242295e-03 * DAYS_PER_YEAR,
             mutable vz: -9.51592254519715870e-05 * DAYS_PER_YEAR,
             mass: 5.15138902046611451e-05 * SOLAR_MASS};
    }

    fn sun() -> Body::props {
        ret {mutable x: 0.0,
             mutable y: 0.0,
             mutable z: 0.0,
             mutable vx: 0.0,
             mutable vy: 0.0,
             mutable vz: 0.0,
             mass: SOLAR_MASS};
    }

    fn offsetMomentum(props: Body::props, px: float, py: float, pz: float) {
        props.vx = -px / SOLAR_MASS;
        props.vy = -py / SOLAR_MASS;
        props.vz = -pz / SOLAR_MASS;
    }

}
