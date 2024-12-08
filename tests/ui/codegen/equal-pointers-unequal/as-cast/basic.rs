//@ known-bug: #107975
//@ compile-flags: -Copt-level=2
//@ run-pass

fn main() {
    let a: usize = {
        let v = 0u8;
        &v as *const _ as usize
    };
    let b: usize = {
        let v = 0u8;
        &v as *const _ as usize
    };

    // `a` and `b` are not equal.
    assert_ne!(a, b);
    // But they are the same number.
    assert_eq!(format!("{a}"), format!("{b}"));
    // And they are equal.
    assert_eq!(a, b);
}
