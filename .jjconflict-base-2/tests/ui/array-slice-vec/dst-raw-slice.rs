// Test bounds checking for DST raw slices

//@ run-fail
//@ error-pattern:index out of bounds
//@ needs-subprocess

#[allow(unconditional_panic)]
fn main() {
    let a: *const [_] = &[1, 2, 3];
    unsafe {
        let _b = (*a)[3];
    }
}
