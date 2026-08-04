use core::fmt::NumBuffer;
use std::mem::size_of_val;

// This test ensures that the `NumBuffer` size and its buffer length doesn't change through
// conversions.
#[test]
fn test_numberbuffer_size() {
    let mut x = NumBuffer::<u32>::new();
    let x_size = size_of_val(&x);
    let y = x.const_cast_into::<u16>();
    let y_size = size_of_val(y);
    // Should work since we come from a bigger buffer originally.
    let z = y.cast_into::<u32>().unwrap();
    let z_size = size_of_val(z);

    assert_eq!(x_size, y_size);
    assert_eq!(x_size, z_size);

    let x = NumBuffer::<u64>::new();
    assert!(x_size < size_of_val(&x));
}
