#![feature(crate_local_distributed_slice)]

#[distributed_slice]
//~^ ERROR `#[distributed_slice]` must take one parameter `crate
const MEOWS: [&str; _];

distributed_slice_element!(MEOWS, "mrow");
distributed_slice_element!(MEOWS, "mew");

fn main() {
    println!("{MEOWS:?}");
}
