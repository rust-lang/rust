// run-rustfix
#![warn(clippy::suboptimal_flops)]

fn main() {
    let one = 1;
    let x = 3f32;

    let y = 4f32;
    let _ = x.powi(2) + y;
    let _ = x + y.powi(2);
    let _ = x + (y as f32).powi(2);
    let _ = (x.powi(2) + y).sqrt();
    let _ = (x + y.powi(2)).sqrt();
    // Cases where the lint shouldn't be applied
    let _ = x.powi(2);
    let _ = x.powi(1 + 1);
    let _ = x.powi(3);
    let _ = x.powi(4) + y;
    let _ = x.powi(one + 1);
    let _ = (x.powi(2) + y.powi(2)).sqrt();
}
