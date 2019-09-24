// run-pass


use std::mem::size_of;

#[cfg(target_pointer_width = "32")]
pub fn main() {
    assert_eq!(size_of::<[u8; (1 << 31) - 1]>(), (1 << 31) - 1);
}

#[cfg(target_pointer_width = "64")]
pub fn main() {
    assert_eq!(size_of::<[u8; (1 << 47) - 1]>(), (1 << 47) - 1);
}
