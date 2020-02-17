#![warn(clippy::suboptimal_flops)]

fn main() {
    let x = 3f32;
    let _ = 2f32.powf(x);
    let _ = 2f32.powf(3.1);
    let _ = 2f32.powf(-3.1);
    let _ = std::f32::consts::E.powf(x);
    let _ = std::f32::consts::E.powf(3.1);
    let _ = std::f32::consts::E.powf(-3.1);
    let _ = x.powf(1.0 / 2.0);
    let _ = x.powf(1.0 / 3.0);
    let _ = x.powf(2.0);
    let _ = x.powf(-2.0);
    let _ = x.powf(2.1);
    let _ = x.powf(-2.1);

    let x = 3f64;
    let _ = 2f64.powf(x);
    let _ = 2f64.powf(3.1);
    let _ = 2f64.powf(-3.1);
    let _ = std::f64::consts::E.powf(x);
    let _ = std::f64::consts::E.powf(3.1);
    let _ = std::f64::consts::E.powf(-3.1);
    let _ = x.powf(1.0 / 2.0);
    let _ = x.powf(1.0 / 3.0);
    let _ = x.powf(2.0);
    let _ = x.powf(-2.0);
    let _ = x.powf(2.1);
    let _ = x.powf(-2.1);
}
