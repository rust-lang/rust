trait FromUnchecked {
    unsafe fn from_unchecked();
}

impl FromUnchecked for [u8; 1] {
    unsafe fn from_unchecked() {
        #[allow(deprecated)]
        let mut array: Self = std::mem::uninitialized();
        let _ptr = &mut array as *mut [u8] as *mut u8;
    }
}

fn main() {
}
