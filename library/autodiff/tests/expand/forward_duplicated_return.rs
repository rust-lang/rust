use autodiff::autodiff;

#[autodiff(d_square2, Forward, Duplicated)]
fn square2(#[dup] a: &Vec<f32>, #[dup] b: &Vec<f32>) -> f32 {
    a.into_iter().map(f32::square).sum()
}
