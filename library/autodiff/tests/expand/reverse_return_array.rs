use autodiff::autodiff;

#[autodiff(d_array, Reverse, Active)]
fn array(#[dup] arr: &[[[f32; 2]; 2]; 2]) -> f32 {
    arr[0][0][0] * arr[1][1][1]
}
