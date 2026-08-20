#![warn(clippy::imprecise_flops)]

macro_rules! m {
    () => {
        std::hint::black_box(1f32)
    };
}

fn main() {
    let x = 3f32;
    let y = 4f32;
    let _ = (x * x + y * y).sqrt();
    //~^ imprecise_flops
    let _ = ((x + 1f32) * (x + 1f32) + y * y).sqrt();
    //~^ imprecise_flops
    let _ = (x.powi(2) + y.powi(2)).sqrt();
    //~^ imprecise_flops
    let _ = x.mul_add(x, y * y).sqrt();
    //~^ imprecise_flops
    // Cases where the lint shouldn't be applied
    let _ = (x * 4f32 + y * y).sqrt();
    let _ = x.mul_add(x, y * y);
    let _ = m!().mul_add(m!(), y * y).sqrt();
}
