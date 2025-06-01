#![feature(crate_local_distributed_slice)]
//@ run-pass
//@ check-run-results

#[distributed_slice(crate)]
static MEOWS: [&str; _];

distributed_slice_element!(MEOWS, "mrow");
distributed_slice_element!(MEOWS, "mew");

fn main() {
    println!("{MEOWS:?}");
}
