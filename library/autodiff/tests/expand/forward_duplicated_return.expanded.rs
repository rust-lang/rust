use autodiff::autodiff;
#[autodiff_into]
fn square2(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    a.into_iter().map(f32::square).sum()
}
#[autodiff_into(Forward, Duplicated, Duplicated, Duplicated)]
fn d_square2(
    a: &Vec<f32>,
    dual_a: &Vec<f32>,
    b: &Vec<f32>,
    dual_b: &Vec<f32>,
) -> (f32, f32, f32) {
    core::hint::black_box((square2(a, b), dual_a, dual_b));
    core::hint::black_box(unsafe { core::mem::zeroed() })
}
