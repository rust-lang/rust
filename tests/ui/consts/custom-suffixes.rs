//@ run-pass

// This tests different kinds of valid suffixes.

fn main() {
    const _A: f64 = 1.;
    const _B: f64 = 1f64;
    const _C: f64 = 1.0f64;
    const _D: f64 = 1e6;
    const _E: f64 = 1.0e9;
    const _F: f64 = 1e-6;
    const _G: f64 = 1.0e-6;
    const _H: f64 = 1.0e06;
    const _I: f64 = 1.0e+6;
    const _J: f64 = 1.0e-6;
    // these ones are perhaps more suprising.
    const _K: f64 = 1.0e0________________________6;
    const _L: f64 = 1.0e________________________6;
    const _M: f64 = 1.0e+________________________6;
    const _N: f64 = 1.0e-________________________6;
    const _O: f64 = 1.0e-________________________9;
}
