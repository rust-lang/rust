//@ run-pass

// After the work to reoptimize structs, it became possible for immediate logic to fail.
// This test verifies that it actually works.

fn main() {
    let c = |a: u8, b: u16, c: u8| {
        assert_eq!(a, 1);
        assert_eq!(b, 2);
        assert_eq!(c, 3);
    };
    c(1, 2, 3);
}
