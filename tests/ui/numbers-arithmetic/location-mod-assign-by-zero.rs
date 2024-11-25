//@ run-fail
//@ check-run-results:location-mod-assign-by-zero.rs

fn main() {
    let mut a = 1;
    a %= &0;
}
