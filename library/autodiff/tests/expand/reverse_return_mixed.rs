use autodiff::autodiff;

#[autodiff(d_sqrt, Reverse, Active)]
fn sqrt(#[active] a: f32, #[dup] b: &f32, c: &f32, #[active] d: f32) -> f32 {
    a * (b * b + c*c*d*d).sqrt()
}
