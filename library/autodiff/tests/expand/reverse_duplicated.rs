use autodiff::autodiff;

#[autodiff(d_square, Reverse, Const)]
fn square(#[dup] a: &Vec<f32>, #[dup] b: &mut f32) {
    *b = a.into_iter().map(f32::square).sum();
}
