#![feature(crate_local_distributed_slice)]

#[distributed_slice(crate)]
const MEOWS: [&str; _];

distributed_slice_element!(MEOWS, "mrow");

const NON_ARRAY: &str = "meow";
distributed_slice_elements!(MEOWS, NON_ARRAY);
//~^ ERROR mismatched types [E0308]


fn main() {
    println!("{MEOWS:?}");
}
