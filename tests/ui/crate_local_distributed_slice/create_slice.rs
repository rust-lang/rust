#![feature(crate_local_distributed_slice)]
// @build-pass

#[distributed_slice(crate)]
const MEOWS: [&str; _];

distributed_slice_element!(MEOWS, "mrow");

fn main() {
    println!("{MEOWS:?}");
}
