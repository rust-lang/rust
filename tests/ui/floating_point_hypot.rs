#![warn(clippy::imprecise_flops)]

#[derive(Clone, Copy, Debug)]
struct CustomMulAdd;
impl CustomMulAdd {
    fn mul_add(self, _a: CustomMulAdd, _b: f32) -> f32 {
        42.0
    }
}

macro_rules! m {
    () => {
        std::hint::black_box(1f32)
    };
}

fn main() {
    let x = 3f32;
    let y = 4f32;
    let custom_mul_add = CustomMulAdd;

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
    // Should not lint: `self_arg` is not a floating point type,
    // even though the call's return type is.
    let _ = custom_mul_add.mul_add(custom_mul_add, y * y).sqrt();
}
