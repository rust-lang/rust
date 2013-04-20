use core::from_str::FromStr;
use core::uint::range;
use core::unstable::intrinsics::sqrtf64;

static PI: f64 = 3.141592653589793;
static SOLAR_MASS: f64 = 4.0 * PI * PI;
static YEAR: f64 = 365.24;
static N_BODIES: uint = 5;

static BODIES: [Planet, ..N_BODIES] = [
    // Sun
    Planet {
        x: [ 0.0, 0.0, 0.0 ],
        v: [ 0.0, 0.0, 0.0 ],
        mass: SOLAR_MASS,
    },
    // Jupiter
    Planet {
        x: [
            4.84143144246472090e+00,
            -1.16032004402742839e+00,
            -1.03622044471123109e-01,
        ],
        v: [
            1.66007664274403694e-03 * YEAR,
            7.69901118419740425e-03 * YEAR,
            -6.90460016972063023e-05 * YEAR,
        ],
        mass: 9.54791938424326609e-04 * SOLAR_MASS,
    },
    // Saturn
    Planet {
        x: [
            8.34336671824457987e+00,
            4.12479856412430479e+00,
            -4.03523417114321381e-01,
        ],
        v: [
            -2.76742510726862411e-03 * YEAR,
            4.99852801234917238e-03 * YEAR,
            2.30417297573763929e-05 * YEAR,
        ],
        mass: 2.85885980666130812e-04 * SOLAR_MASS,
    },
    // Uranus
    Planet {
        x: [
            1.28943695621391310e+01,
            -1.51111514016986312e+01,
            -2.23307578892655734e-01,
        ],
        v: [
            2.96460137564761618e-03 * YEAR,
            2.37847173959480950e-03 * YEAR,
            -2.96589568540237556e-05 * YEAR,
        ],
        mass: 4.36624404335156298e-05 * SOLAR_MASS,
    },
    // Neptune
    Planet {
        x: [
            1.53796971148509165e+01,
            -2.59193146099879641e+01,
            1.79258772950371181e-01,
        ],
        v: [
            2.68067772490389322e-03 * YEAR,
            1.62824170038242295e-03 * YEAR,
            -9.51592254519715870e-05 * YEAR,
        ],
        mass: 5.15138902046611451e-05 * SOLAR_MASS,
    },
];

struct Planet {
    x: [f64, ..3],
    v: [f64, ..3],
    mass: f64,
}

fn advance(bodies: &mut [Planet, ..N_BODIES], dt: f64, steps: i32) {
    let mut d = [ 0.0, ..3 ];
    for (steps as uint).times {
        for range(0, N_BODIES) |i| {
            for range(i + 1, N_BODIES) |j| {
                d[0] = bodies[i].x[0] - bodies[j].x[0];
                d[1] = bodies[i].x[1] - bodies[j].x[1];
                d[2] = bodies[i].x[2] - bodies[j].x[2];

                let d2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
                let mag = dt / (d2 * sqrtf64(d2));

                let a_mass = bodies[i].mass, b_mass = bodies[j].mass;
                bodies[i].v[0] -= d[0] * b_mass * mag;
                bodies[i].v[1] -= d[1] * b_mass * mag;
                bodies[i].v[2] -= d[2] * b_mass * mag;

                bodies[j].v[0] += d[0] * a_mass * mag;
                bodies[j].v[1] += d[1] * a_mass * mag;
                bodies[j].v[2] += d[2] * a_mass * mag;
            }
        }

        for vec::each_mut(*bodies) |a| {
            a.x[0] += dt * a.v[0];
            a.x[1] += dt * a.v[1];
            a.x[2] += dt * a.v[2];
        }
    }
}

fn energy(bodies: &[Planet, ..N_BODIES]) -> f64 {
    let mut e = 0.0;
    let mut d = [ 0.0, ..3 ];
    for range(0, N_BODIES) |i| {
        for range(0, 3) |k| {
            e += bodies[i].mass * bodies[i].v[k] * bodies[i].v[k] / 2.0;
        }

        for range(i + 1, N_BODIES) |j| {
            for range(0, 3) |k| {
                d[k] = bodies[i].x[k] - bodies[j].x[k];
            }
            let dist = sqrtf64(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
            e -= bodies[i].mass * bodies[j].mass / dist;
        }
    }
    e
}

fn offset_momentum(bodies: &mut [Planet, ..N_BODIES]) {
    for range(0, N_BODIES) |i| {
        for range(0, 3) |k| {
            bodies[0].v[k] -= bodies[i].v[k] * bodies[i].mass / SOLAR_MASS;
        }
    }
}

fn main() {
    let n: i32 = FromStr::from_str(os::args()[1]).get();
    let mut bodies = BODIES;

    offset_momentum(&mut bodies);
    println(fmt!("%.9f", energy(&bodies) as float));

    advance(&mut bodies, 0.01, n);

    println(fmt!("%.9f", energy(&bodies) as float));
}

