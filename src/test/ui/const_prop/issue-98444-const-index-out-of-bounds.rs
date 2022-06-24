// build-fail
// Need to use build-fail because check doesn't run constant propagation.

fn main() {
    let xs: [i32; 5] = [1, 2, 3, 4, 5];
    let _ = &xs;
    let _ = xs[7]; //~ ERROR this operation will panic at runtime
}
