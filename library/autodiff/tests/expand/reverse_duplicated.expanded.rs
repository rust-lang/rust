use autodiff::autodiff;
#[autodiff_into]
fn squre(a: &Vec<f32>, b: &mut f32) {
    *b = a.into_iter().map(f32::square).sum();
}
#[autodiff_into(Reverse, Const, Duplicated, Duplicated)]
fn d_square(a: &Vec<f32>, d_a: &mut Vec<f32>, b: &mut f32, adj_b: &f32) {
    std::hint::black_box((squre(a, b), &d_a, &adj_b, &a, &b));
}
