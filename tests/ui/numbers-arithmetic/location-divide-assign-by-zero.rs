//@ run-fail
//@ check-run-results:location-divide-assign-by-zero.rs

fn main() {
    let mut a = 1;
    a /= &0;
}
