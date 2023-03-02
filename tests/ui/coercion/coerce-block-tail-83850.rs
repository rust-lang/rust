// run-pass
fn f(_: &[i32]) {}

fn main() {
    f(&Box::new([1, 2]));
}
