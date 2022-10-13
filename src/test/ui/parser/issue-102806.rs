#![allow(dead_code)]

struct V3 {
    x: f32,
    y: f32,
    z: f32,
}

fn pz(v: V3) {
    let _ = V3 { z: 0.0, ...v};
    //~^ ERROR expected `..`
    //~| ERROR missing fields `x` and `y` in initializer of `V3`
    let _ = V3 { z: 0.0, ... };
    //~^ expected identifier
    //~| ERROR missing fields `x` and `y` in initializer of `V3`

    let _ = V3 { z: 0.0, ...Default::default() };
    //~^ ERROR expected `..`
    //~| ERROR missing fields `x` and `y` in initializer of `V3`
}

fn main() {}
