// check-pass

trait FromUnchecked {
    unsafe fn from_unchecked();
}

impl FromUnchecked for [u8; 1] {
    unsafe fn from_unchecked() {
        let mut array: Self = std::mem::zeroed();
        let _ptr = &mut array as *mut [u8] as *mut u8;
    }
}

fn main() {
}
