//@ run-fail
//@ check-run-results:location-sub-overflow.rs

fn main() {
    let _: u8 = 0 - &1;
}
