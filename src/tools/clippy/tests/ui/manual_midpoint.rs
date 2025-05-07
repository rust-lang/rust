#![warn(clippy::manual_midpoint)]

macro_rules! mac {
    ($a: expr, $b: expr) => {
        ($a + $b) / 2
    };
}

macro_rules! add {
    ($a: expr, $b: expr) => {
        ($a + $b)
    };
}

macro_rules! two {
    () => {
        2
    };
}

#[clippy::msrv = "1.84"]
fn older_msrv() {
    let a: u32 = 10;
    let _ = (a + 5) / 2;
}

#[clippy::msrv = "1.85"]
fn main() {
    let a: u32 = 10;
    let _ = (a + 5) / 2; //~ ERROR: manual implementation of `midpoint`

    let f: f32 = 10.0;
    let _ = (f + 5.0) / 2.0; //~ ERROR: manual implementation of `midpoint`

    let _: u32 = 5 + (8 + 8) / 2 + 2; //~ ERROR: manual implementation of `midpoint`
    let _: u32 = const { (8 + 8) / 2 }; //~ ERROR: manual implementation of `midpoint`
    let _: f64 = const { (8.0f64 + 8.) / 2. }; //~ ERROR: manual implementation of `midpoint`
    let _: u32 = (u32::default() + u32::default()) / 2; //~ ERROR: manual implementation of `midpoint`
    let _: u32 = (two!() + two!()) / 2; //~ ERROR: manual implementation of `midpoint`

    // Do not lint in presence of an addition with more than 2 operands
    let _: u32 = (10 + 20 + 30) / 2;

    // Do not lint if whole or part is coming from a macro
    let _ = mac!(10, 20);
    let _: u32 = add!(10u32, 20u32) / 2;
    let _: u32 = (10 + 20) / two!();

    // Do not lint if a literal is not present
    let _ = (f + 5.0) / (1.0 + 1.0);

    // Do not lint on signed integer types
    let i: i32 = 10;
    let _ = (i + 5) / 2;

    // Do not lint on (x+1)/2 or (1+x)/2 as this looks more like a `div_ceil()` operation
    let _ = (i + 1) / 2;
    let _ = (1 + i) / 2;

    // But if we see (x+1.0)/2.0 or (x+1.0)/2.0, it is probably a midpoint operation
    let _ = (f + 1.0) / 2.0; //~ ERROR: manual implementation of `midpoint`
    let _ = (1.0 + f) / 2.0; //~ ERROR: manual implementation of `midpoint`
}

#[clippy::msrv = "1.86"]
fn older_signed_midpoint(i: i32) {
    // Do not lint
    let _ = (i + 10) / 2;
}

#[clippy::msrv = "1.87"]
fn signed_midpoint(i: i32) {
    let _ = (i + 10) / 2; //~ ERROR: manual implementation of `midpoint`
}
