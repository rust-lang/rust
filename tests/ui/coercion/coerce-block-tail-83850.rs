//@ check-fail
fn f(_: &[i32]) {}

fn main() {
    f(&Box::new([1, 2]));
    //~^ ERROR mismatched types
}
