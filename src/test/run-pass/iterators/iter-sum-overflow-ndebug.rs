// run-pass
// compile-flags: -C debug_assertions=no

fn main() {
    assert_eq!([1i32, i32::max_value()].iter().sum::<i32>(),
               1i32.wrapping_add(i32::max_value()));
    assert_eq!([2i32, i32::max_value()].iter().product::<i32>(),
               2i32.wrapping_mul(i32::max_value()));

    assert_eq!([1i32, i32::max_value()].iter().cloned().sum::<i32>(),
               1i32.wrapping_add(i32::max_value()));
    assert_eq!([2i32, i32::max_value()].iter().cloned().product::<i32>(),
               2i32.wrapping_mul(i32::max_value()));
}
