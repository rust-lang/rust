//@ known-bug: rust-lang/rust#130372

pub fn test_va_copy(_: u64, mut ap: ...) {}

pub fn main() {
    unsafe {
        test_va_copy();

        call(x);
    }
}
