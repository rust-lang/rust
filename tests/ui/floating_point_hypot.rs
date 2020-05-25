// run-rustfix
#![warn(clippy::suboptimal_flops, clippy::imprecise_flops)]

fn main() {
    let x = 3f32;
    let y = 4f32;
    let _ = (x * x + y * y).sqrt();
    let _ = ((x + 1f32) * (x + 1f32) + y * y).sqrt();
    let _ = (x.powi(2) + y.powi(2)).sqrt();
    // Cases where the lint shouldn't be applied
    let _ = x.mul_add(x, y * y).sqrt();
    let _ = (x * 4f32 + y * y).sqrt();
}
