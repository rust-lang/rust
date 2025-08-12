// https://github.com/rust-lang/rust/issues/58212
//@ check-pass

trait FromUnchecked {
    fn from_unchecked();
}

impl FromUnchecked for [u8; 1] {
    fn from_unchecked() {
        let mut array: Self = [0; 1];
        let _ptr = &mut array as *mut [u8] as *mut u8;
    }
}

fn main() {
}
