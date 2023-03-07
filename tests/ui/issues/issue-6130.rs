// run-pass

pub fn main() {
    let i: usize = 0;
    assert!(i <= 0xFFFF_FFFF);

    let i: isize = 0;
    assert!(i >= -0x8000_0000);
    assert!(i <= 0x7FFF_FFFF);
}
