// FIXME(f16_f128): add tests when exp is available

#![warn(clippy::imprecise_flops)]
#![allow(clippy::unnecessary_cast)]

fn main() {
    let x = 2f32;
    let _ = x.exp() - 1.0;
    let _ = x.exp() - 1.0 + 2.0;
    let _ = (x as f32).exp() - 1.0 + 2.0;
    // Cases where the lint shouldn't be applied
    let _ = x.exp() - 2.0;
    let _ = x.exp() - 1.0 * 2.0;

    let x = 2f64;
    let _ = x.exp() - 1.0;
    let _ = x.exp() - 1.0 + 2.0;
    // Cases where the lint shouldn't be applied
    let _ = x.exp() - 2.0;
    let _ = x.exp() - 1.0 * 2.0;
}
