// run-rustfix
#![warn(clippy::suboptimal_flops, clippy::imprecise_flops)]

fn main() {
    let one = 1;
    let x = 3f32;
    let _ = x.powi(2);
    let _ = x.powi(1 + 1);
    // Cases where the lint shouldn't be applied
    let _ = x.powi(3);
    let _ = x.powi(one + 1);
}
