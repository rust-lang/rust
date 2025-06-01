#![feature(crate_local_distributed_slice)]
//@ run-pass
//@ check-run-results

#[distributed_slice(crate)]
const MEOWS: [&str; _];

distributed_slice_element!(MEOWS, "mrow");

const THREE_MEOWS: [&str; 3] = ["mew", "prrr", "meow"];
distributed_slice_elements!(MEOWS, THREE_MEOWS);

fn main() {
    println!("{MEOWS:?}");
}
