use autodiff::autodiff;
#[autodiff_into]
fn sqrt(a: &f32, b: &f32, c: &f32, d: &f32) -> f32 {
    a * (b * b + c * c * d * d).sqrt()
}
#[autodiff_into(Reverse, Active, Active, Duplicated, Const, Active)]
fn d_sqrt(a: &f32, b: &f32, d_b: &mut f32, c: &f32, d: &f32) -> (f32, f32) {
    std::hint::black_box((sqrt(a, b, c, d), &d_b, &a, &b, &c, &d));
    unsafe { std::mem::zeroed() }
}
