#![deny(unreachable_patterns)]
#![feature(f128)]
#![feature(f16)]

fn main() {
    match 0.0 {
        0.0..=1.0 => {}
        _ => {} // ok
    }

    match 0.0 {
        //~^ ERROR non-exhaustive patterns
        0.0..=1.0 => {}
    }

    match 1.0f16 {
        0.01f16..=6.5f16 => {}
        0.01f16 => {} //~ ERROR unreachable pattern
        0.02f16 => {} //~ ERROR unreachable pattern
        6.5f16 => {}  //~ ERROR unreachable pattern
        _ => {}
    };
    match 1.0f16 {
        0.01f16..6.5f16 => {}
        6.5f16 => {} // this is reachable
        _ => {}
    };

    match 1.0f32 {
        0.01f32..=6.5f32 => {}
        0.01f32 => {} //~ ERROR unreachable pattern
        0.02f32 => {} //~ ERROR unreachable pattern
        6.5f32 => {}  //~ ERROR unreachable pattern
        _ => {}
    };
    match 1.0f32 {
        0.01f32..6.5f32 => {}
        6.5f32 => {} // this is reachable
        _ => {}
    };

    match 1.0f64 {
        0.01f64..=6.5f64 => {}
        0.005f64 => {}
        0.01f64 => {} //~ ERROR unreachable pattern
        0.02f64 => {} //~ ERROR unreachable pattern
        6.5f64 => {}  //~ ERROR unreachable pattern
        6.6f64 => {}
        1.0f64..=4.0f64 => {} //~ ERROR unreachable pattern
        5.0f64..=7.0f64 => {}
        _ => {}
    };
    match 1.0f64 {
        0.01f64..6.5f64 => {}
        6.5f64 => {} // this is reachable
        _ => {}
    };

    match 1.0f128 {
        0.01f128..=6.5f128 => {}
        0.005f128 => {}
        0.01f128 => {} //~ ERROR unreachable pattern
        0.02f128 => {} //~ ERROR unreachable pattern
        6.5f128 => {}  //~ ERROR unreachable pattern
        6.6f128 => {}
        1.0f128..=4.0f128 => {} //~ ERROR unreachable pattern
        5.0f128..=7.0f128 => {}
        _ => {}
    };
    match 1.0f128 {
        0.01f128..6.5f128 => {}
        6.5f128 => {} // this is reachable
        _ => {}
    };
}
