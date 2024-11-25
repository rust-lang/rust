//@ run-fail
//@ check-run-results:location-mod-by-zero.rs

fn main() {
    let _ = 1 % &0;
}
