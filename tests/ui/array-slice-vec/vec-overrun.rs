// run-fail
//@error-in-other-file:index out of bounds: the len is 1 but the index is 2
//@ignore-target-emscripten no processes

fn main() {
    let v: Vec<isize> = vec![10];
    let x: usize = 0;
    assert_eq!(v[x], 10);
    // Bounds-check panic.

    assert_eq!(v[x + 2], 20);
}
