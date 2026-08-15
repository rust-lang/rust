//@compile-flags: -Zmiri-strict-provenance

fn main() {
    unsafe {
        let a = [1, 2, 3];
        let s = &a[0..0];
        assert_eq!(s.len(), 0);
        assert_eq!(*s.as_ptr().add(1), 2); //~ ERROR: /retag .* tag does not exist in the borrow stack/
    }
}
