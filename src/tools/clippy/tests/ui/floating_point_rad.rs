#![warn(clippy::suboptimal_flops)]

/// Allow suboptimal_flops in constant context
pub const fn const_context() {
    let x = 3f32;
    let _ = x * 180f32 / std::f32::consts::PI;
}

pub fn issue9391(degrees: i64) {
    let _ = degrees as f64 * std::f64::consts::PI / 180.0;
    //~^ suboptimal_flops
    let _ = degrees as f64 * 180.0 / std::f64::consts::PI;
    //~^ suboptimal_flops
}

fn main() {
    let x = 3f32;
    let _ = x * 180f32 / std::f32::consts::PI;
    //~^ suboptimal_flops
    let _ = 90. * 180f64 / std::f64::consts::PI;
    //~^ suboptimal_flops
    let _ = 90.5 * 180f64 / std::f64::consts::PI;
    //~^ suboptimal_flops
    let _ = x * std::f32::consts::PI / 180f32;
    //~^ suboptimal_flops
    let _ = 90. * std::f32::consts::PI / 180f32;
    //~^ suboptimal_flops
    let _ = 90.5 * std::f32::consts::PI / 180f32;
    //~^ suboptimal_flops
    // let _ = 90.5 * 80. * std::f32::consts::PI / 180f32;
    // Cases where the lint shouldn't be applied
    let _ = x * 90f32 / std::f32::consts::PI;
    let _ = x * std::f32::consts::PI / 90f32;
    let _ = x * 180f32 / std::f32::consts::E;
    let _ = x * std::f32::consts::E / 180f32;
}
