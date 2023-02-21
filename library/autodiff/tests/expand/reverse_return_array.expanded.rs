use autodiff::autodiff;
#[autodiff_into]
fn array(arr: &[[[f32; 2]; 2]; 2]) -> f32 {
    arr[0][0][0] * arr[1][1][1]
}
#[autodiff_into(Reverse, Active, Duplicated)]
fn d_array(arr: &[[[f32; 2]; 2]; 2], d_arr: &mut [[[f32; 2]; 2]; 2], ret_adj: f32) {
    std::hint::black_box((array(arr), &d_arr, &ret_adj, &arr));
}
