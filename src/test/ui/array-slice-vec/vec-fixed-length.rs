// run-pass


use std::mem::size_of;

#[cfg(not(target_pointer_width = "64"))]
fn test_big_vec() {}

#[cfg(target_pointer_width = "64")]
fn test_big_vec()
{
    assert_eq!(size_of::<[u8; (1 << 32)]>(), (1 << 32));
}

fn main() {
    let x: [isize; 4] = [1, 2, 3, 4];
    assert_eq!(x[0], 1);
    assert_eq!(x[1], 2);
    assert_eq!(x[2], 3);
    assert_eq!(x[3], 4);

    assert_eq!(size_of::<[u8; 4]>(), 4);
    test_big_vec();
}
