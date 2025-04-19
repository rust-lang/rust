#![allow(dead_code)]

#[derive(Default)]
struct V3 {
    x: f32,
    y: f32,
    z: f32,
}

fn pz(v: V3) {
    let _ = V3 { z: 0.0, ...v};
    //~^ ERROR expected `..`

    let _ = V3 { z: 0.0, ...Default::default() };
    //~^ ERROR expected `..`

    let _ = V3 { z: 0.0, ... };
    //~^ ERROR expected identifier
    //~| ERROR missing fields `x` and `y` in initializer of `V3`

    let V3 { z: val, ... } = v;
    //~^ ERROR expected field pattern
}

fn main() {}
