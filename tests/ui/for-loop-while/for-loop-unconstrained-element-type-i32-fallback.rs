//@ run-pass
// Test that the type of `sum` falls back to `i32` here,
// and that the for loop desugaring doesn't interfere with
// that.

fn main() {
    let mut sum = 0;
    for i in Vec::new() {
        sum += &i;
    }
}
