#![feature(crate_local_distributed_slice)]
//@ run-pass
//@ check-run-results

#[distributed_slice(crate)]
const MEOWS: [&str; _];

const NYA: &str = "nya";

distributed_slice_element!(MEOWS, "mrow");
distributed_slice_element!(MEOWS, NYA);

fn main() {
    println!("{MEOWS:?}");
}
