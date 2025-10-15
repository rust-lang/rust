#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[repr(C)]
struct Container {
    header: i64,
    data: [f32; 1000],
}

#[autodiff_reverse(d_test, Duplicated, Active)]
#[no_mangle]
#[inline(never)]
fn test_mixed_struct(container: &Container) -> f32 {
    container.data[0] + container.data[999]
}

fn main() {
    let container = Container { header: 42, data: [1.0; 1000] };
    let mut d_container = Container { header: 0, data: [0.0; 1000] };
    let result = d_test(&container, &mut d_container, 1.0);
    std::hint::black_box(result);
}
