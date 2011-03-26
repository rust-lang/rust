// xfail-boot
// based on:
// http://shootout.alioth.debian.org/u32/benchmark.php?test=nbody&lang=java

fn main() {

    let vec[int] inputs = vec(
        50000
        //these segfault :(
        //500000,
        //5000000,
        //50000000
        );

    let vec[Body.props] bodies = NBodySystem.MakeNBodySystem();

    for (int n in inputs) {
        log NBodySystem.energy(bodies);

        let int i = 0;
        while (i < n) {
            NBodySystem.advance(bodies, 0.01);
            i += 1;
        }
        log NBodySystem.energy(bodies);
    }
}

// making a native call to sqrt
native "rust" mod rustrt {
    fn squareroot(&float input, &mutable float output);
}

// Body.props is a record of floats, so
// vec[Body.props] is a vector of records of floats

mod NBodySystem {

    fn MakeNBodySystem() -> vec[Body.props] {
        let vec[Body.props] bodies = vec(
            // these each return a Body.props
            Body.sun(), 
            Body.jupiter(), 
            Body.saturn(), 
            Body.uranus(), 
            Body.neptune());

        let float px = 0.0;
        let float py = 0.0;
        let float pz = 0.0;

        let int i = 0;
        while (i < 5) {
            px += bodies.(i).vx * bodies.(i).mass;
            py += bodies.(i).vy * bodies.(i).mass;
            pz += bodies.(i).vz * bodies.(i).mass;

            i += 1;
        }

        // side-effecting
        Body.offsetMomentum(bodies.(0), px, py, pz);

        ret bodies;
    }

    fn advance(vec[Body.props] bodies, float dt) -> () {

        let int i = 0;
        while (i < 5) {
            let int j = i+1;
            while (j < 5) {
                let float dx = bodies.(i).x - bodies.(j).x;
                let float dy = bodies.(i).y - bodies.(j).y;
                let float dz = bodies.(i).z - bodies.(j).z;

                let float dSquared = dx * dx + dy * dy + dz * dz;

                let float distance;
                rustrt.squareroot(dSquared, distance);
                let float mag = dt / (dSquared * distance);

                bodies.(i).vx -= dx * bodies.(j).mass * mag;
                bodies.(i).vy -= dy * bodies.(j).mass * mag;
                bodies.(i).vz -= dz * bodies.(j).mass * mag;

                bodies.(j).vx += dx * bodies.(i).mass * mag;
                bodies.(j).vy += dy * bodies.(i).mass * mag;
                bodies.(j).vz += dz * bodies.(i).mass * mag;

                j += 1;

            }

            i += 1;
        }

        i = 0;
        while (i < 5) {

            bodies.(i).x += dt * bodies.(i).vx;
            bodies.(i).y += dt * bodies.(i).vy;
            bodies.(i).z += dt * bodies.(i).vz;

            i += 1;
        }
    }

    fn energy(vec[Body.props] bodies) -> float {
        let float dx;
        let float dy;
        let float dz;
        let float distance;
        let float e = 0.0;

        let int i = 0;
        while (i < 5) {
            e += 0.5 * bodies.(i).mass *
                (  bodies.(i).vx * bodies.(i).vx
                 + bodies.(i).vy * bodies.(i).vy
                 + bodies.(i).vz * bodies.(i).vz );

            let int j = i+1;
            while (j < 5) {
                dx = bodies.(i).x - bodies.(j).x;
                dy = bodies.(i).y - bodies.(j).y;
                dz = bodies.(i).z - bodies.(j).z;

                rustrt.squareroot(dx*dx + dy*dy + dz*dz, distance);
                e -= (bodies.(i).mass * bodies.(j).mass) / distance;
                
                j += 1;
            }

            i += 1;
        }
        ret e;

    }
}

mod Body {

    const float PI = 3.141592653589793;
    const float SOLAR_MASS = 39.478417604357432; // was 4 * PI * PI originally
    const float DAYS_PER_YEAR = 365.24;

    type props = rec(mutable float x, 
                     mutable float y, 
                     mutable float z, 
                     mutable float vx, 
                     mutable float vy,
                     mutable float vz, 
                     float mass);

    fn jupiter() -> Body.props {
        ret rec(
            mutable x  =  4.84143144246472090e+00,
            mutable y  = -1.16032004402742839e+00,
            mutable z  = -1.03622044471123109e-01,
            mutable vx =  1.66007664274403694e-03 * DAYS_PER_YEAR,
            mutable vy =  7.69901118419740425e-03 * DAYS_PER_YEAR,
            mutable vz = -6.90460016972063023e-05 * DAYS_PER_YEAR,
            mass       =  9.54791938424326609e-04 * SOLAR_MASS
            );
    }

    fn saturn() -> Body.props {
        ret rec(
            mutable x  =  8.34336671824457987e+00,
            mutable y  =  4.12479856412430479e+00,
            mutable z  = -4.03523417114321381e-01,
            mutable vx = -2.76742510726862411e-03 * DAYS_PER_YEAR,
            mutable vy =  4.99852801234917238e-03 * DAYS_PER_YEAR,
            mutable vz =  2.30417297573763929e-05 * DAYS_PER_YEAR,
            mass       =  2.85885980666130812e-04 * SOLAR_MASS
            );
   }

    fn uranus() -> Body.props {
        ret rec(
            mutable x  =  1.28943695621391310e+01,
            mutable y  = -1.51111514016986312e+01,
            mutable z  = -2.23307578892655734e-01,
            mutable vx =  2.96460137564761618e-03 * DAYS_PER_YEAR,
            mutable vy =  2.37847173959480950e-03 * DAYS_PER_YEAR,
            mutable vz = -2.96589568540237556e-05 * DAYS_PER_YEAR,
            mass       =  4.36624404335156298e-05 * SOLAR_MASS
            );
    }

    fn neptune() -> Body.props {
        ret rec(
            mutable x  =  1.53796971148509165e+01,
            mutable y  = -2.59193146099879641e+01,
            mutable z  =  1.79258772950371181e-01,
            mutable vx =  2.68067772490389322e-03 * DAYS_PER_YEAR,
            mutable vy =  1.62824170038242295e-03 * DAYS_PER_YEAR,
            mutable vz = -9.51592254519715870e-05 * DAYS_PER_YEAR,
            mass       =  5.15138902046611451e-05 * SOLAR_MASS
            );
   }

   fn sun() -> Body.props {
       ret rec(
           mutable x  =  0.0,
           mutable y  =  0.0,
           mutable z  =  0.0,
           mutable vx =  0.0,
           mutable vy =  0.0,
           mutable vz =  0.0,
           mass       =  SOLAR_MASS
           );
   }

   impure fn offsetMomentum(&Body.props props,
                            float px, 
                            float py, 
                            float pz) -> () {
       props.vx = -px / SOLAR_MASS;
       props.vy = -py / SOLAR_MASS;
       props.vz = -pz / SOLAR_MASS;
   }

}
