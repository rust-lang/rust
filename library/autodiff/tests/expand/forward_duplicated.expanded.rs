use autodiff::autodiff;
#[autodiff_into]
fn square(a: &Vec<f32>, b: &mut f32) {
    *b = a.into_iter().map(f32::square).sum();
}
#[autodiff_into(Forward, Const, Duplicated, Duplicated)]
fn d_square(a: &Vec<f32>, dual_a: &Vec<f32>, b: &mut f32, grad_b: &mut f32) {
    core::hint::black_box((square(a, b), dual_a, grad_b));
    core::hint::black_box(unsafe { core::mem::zeroed() })
}
