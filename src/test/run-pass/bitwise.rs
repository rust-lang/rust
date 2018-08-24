#[cfg(any(target_pointer_width = "32"))]
fn target() {
    assert_eq!(-1000isize as usize >> 3_usize, 536870787_usize);
}

#[cfg(any(target_pointer_width = "64"))]
fn target() {
    assert_eq!(-1000isize as usize >> 3_usize, 2305843009213693827_usize);
}

fn general() {
    let mut a: isize = 1;
    let mut b: isize = 2;
    a ^= b;
    b ^= a;
    a = a ^ b;
    println!("{}", a);
    println!("{}", b);
    assert_eq!(b, 1);
    assert_eq!(a, 2);
    assert_eq!(!0xf0_isize & 0xff, 0xf);
    assert_eq!(0xf0_isize | 0xf, 0xff);
    assert_eq!(0xf_isize << 4, 0xf0);
    assert_eq!(0xf0_isize >> 4, 0xf);
    assert_eq!(-16 >> 2, -4);
    assert_eq!(0b1010_1010_isize | 0b0101_0101, 0xff);
}

pub fn main() {
    general();
    target();
}
