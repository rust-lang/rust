// run-rustfix
#![warn(clippy::suboptimal_flops)]

fn main() {
    let x = 3f32;
    let _ = x * 180f32 / std::f32::consts::PI;
    let _ = x * std::f32::consts::PI / 180f32;
    // Cases where the lint shouldn't be applied
    let _ = x * 90f32 / std::f32::consts::PI;
    let _ = x * std::f32::consts::PI / 90f32;
    let _ = x * 180f32 / std::f32::consts::E;
    let _ = x * std::f32::consts::E / 180f32;
}
