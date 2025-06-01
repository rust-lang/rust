#![feature(crate_local_distributed_slice)]
//@ run-pass
//@ check-run-results

#[distributed_slice(crate)]
const MEOWS: [&str; _];

distributed_slice_element!(MEOWS, "mrow");
distributed_slice_elements!(MEOWS, ["mew", "prrr", "meow"]);

fn main() {
    println!("{MEOWS:?}");
}
