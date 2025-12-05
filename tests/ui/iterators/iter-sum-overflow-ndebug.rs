//@ run-pass
//@ compile-flags: -C debug_assertions=no

fn main() {
    assert_eq!([1i32, i32::MAX].iter().sum::<i32>(),
               1i32.wrapping_add(i32::MAX));
    assert_eq!([2i32, i32::MAX].iter().product::<i32>(),
               2i32.wrapping_mul(i32::MAX));

    assert_eq!([1i32, i32::MAX].iter().cloned().sum::<i32>(),
               1i32.wrapping_add(i32::MAX));
    assert_eq!([2i32, i32::MAX].iter().cloned().product::<i32>(),
               2i32.wrapping_mul(i32::MAX));
}
