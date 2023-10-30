use autodiff::autodiff;
#[autodiff_into]
fn array(arr: &[[[f32; 2]; 2]; 2]) -> f32 {
    arr[0][0][0] * arr[1][1][1]
}
#[autodiff_into(Reverse, Active, Duplicated)]
fn d_array(arr: &[[[f32; 2]; 2]; 2], grad_arr: &mut [[[f32; 2]; 2]; 2], tang_y: f32) {
    std::hint::black_box((array(arr), grad_arr, tang_y));
    std::hint::black_box(unsafe { std::mem::zeroed() })
}
