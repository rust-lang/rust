// based on:
// http://shootout.alioth.debian.org/u32/benchmark.php?test=nbody&lang=java

fn main() {

    log "foo";

    let vec[int] inputs = vec(
        50000
        //these segfault :(
        //500000,
        //5000000,
        //50000000
        );

    let vec[Body.props] bodies = NBodySystem.MakeNBodySystem();

    for (int n in inputs) {
        // TODO: make #fmt handle floats?
        log NBodySystem.energy(bodies);

        let int i = 0;
        while (i < n) {
            bodies = NBodySystem.advance(bodies, 0.01);
            i = i+1;
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
        // can't iterate over a record?  how about a vector, then?
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

        for (Body.props body in bodies) {
            px += body.vx * body.mass;
            py += body.vy * body.mass;
            pz += body.vz * body.mass;
        }

        // side-effecting
        Body.offsetMomentum(bodies.(0), px, py, pz);

        ret bodies;
    }

    fn advance(vec[Body.props] bodies, float dt) -> vec[Body.props] {
        for (Body.props ibody in bodies) {

            let Body.props iBody = ibody;

            for (Body.props jbody in bodies) {
                let float dx = iBody.x - jbody.x;
                let float dy = iBody.y - jbody.y;
                let float dz = iBody.z - jbody.z;

                let float dSquared = dx * dx + dy * dy + dz * dz;

                let float distance;
                rustrt.squareroot(dSquared, distance);
                let float mag = dt / (dSquared * distance);

                iBody.vx -= dx * jbody.mass * mag;
                iBody.vy -= dy * jbody.mass * mag;
                iBody.vz -= dz * jbody.mass * mag;

                jbody.vx += dx * iBody.mass * mag;
                jbody.vy += dy * iBody.mass * mag;
                jbody.vz += dz * iBody.mass * mag;
            }
        }        

        for (Body.props body in bodies) {
            body.x += dt * body.vx;
            body.y += dt * body.vy;
            body.z += dt * body.vz;
        }

        ret bodies;
    }

    fn energy(vec[Body.props] bodies) -> float {
        let float dx;
        let float dy;
        let float dz;
        let float distance;
        let float e = 0.0;

        for (Body.props ibody in bodies) {

            // do we need this?
            let Body.props iBody = ibody;

            e += 0.5 * iBody.mass *
                ( iBody.vx * iBody.vx
                  + iBody.vy * iBody.vy
                  + iBody.vz * iBody.vz );

            for (Body.props jbody in bodies) {

                // do we need this?
                let Body.props jBody = jbody;

                dx = iBody.x - jBody.x;
                dy = iBody.y - jBody.y;
                dz = iBody.z - jBody.z;

                rustrt.squareroot(dx*dx + dy*dy + dz*dz, distance);
                e -= (iBody.mass * jBody.mass) / distance;
            }
        }
        ret e;
    }

}

mod Body {
    
    const float PI = 3.14;
    const float SOLAR_MASS = 39.47; // was 4 * PI * PI originally
    const float DAYS_PER_YEAR = 365.24;

    type props = rec(float x, 
                     float y, 
                     float z, 
                     float vx, 
                     float vy,
                     float vz, 
                     float mass);

    fn jupiter() -> Body.props {
        // current limitation of the float lexer: decimal part has to
        // fit into a 32-bit int.
        
        let Body.props p;
        p.x    =  4.84e+00;
        p.y    = -1.16e+00;
        p.z    = -1.03e-01;
        p.vx   =  1.66e-03 * DAYS_PER_YEAR;
        p.vy   =  7.69e-03 * DAYS_PER_YEAR;
        p.vz   = -6.90e-05 * DAYS_PER_YEAR;
        p.mass =  9.54e-04 * SOLAR_MASS;
        ret p;
    }

    fn saturn() -> Body.props {
        let Body.props p;
        p.x    =  8.34e+00;
        p.y    =  4.12e+00;
        p.z    = -4.03e-01;
        p.vx   = -2.76e-03 * DAYS_PER_YEAR;
        p.vy   =  4.99e-03 * DAYS_PER_YEAR;
        p.vz   =  2.30e-05 * DAYS_PER_YEAR;
        p.mass =  2.85e-04 * SOLAR_MASS;
        ret p;
   }

    fn uranus() -> Body.props {
        let Body.props p;
        p.x    =  1.28e+01;
        p.y    = -1.51e+01;
        p.z    = -2.23e-01;
        p.vx   =  2.96e-03 * DAYS_PER_YEAR;
        p.vy   =  2.37e-03 * DAYS_PER_YEAR;
        p.vz   = -2.96e-05 * DAYS_PER_YEAR;
        p.mass =  4.36e-05 * SOLAR_MASS;
        ret p;
    }

    fn neptune() -> Body.props {
        let Body.props p;
        p.x    =  1.53e+01;
        p.y    = -2.59e+01;
        p.z    =  1.79e-01;
        p.vx   =  2.68e-03 * DAYS_PER_YEAR;
        p.vy   =  1.62e-03 * DAYS_PER_YEAR;
        p.vz   = -9.51e-05 * DAYS_PER_YEAR;
        p.mass =  5.15e-05 * SOLAR_MASS;
        ret p;
   }

   fn sun() -> Body.props {
        let Body.props p;
        p.mass = SOLAR_MASS;
        ret p;
   }

   impure fn offsetMomentum(mutable Body.props props,
                     float px, 
                     float py, 
                     float pz) -> Body.props {

       props.vx = -px / SOLAR_MASS;
       props.vy = -py / SOLAR_MASS;
       props.vz = -pz / SOLAR_MASS;
   }

}
