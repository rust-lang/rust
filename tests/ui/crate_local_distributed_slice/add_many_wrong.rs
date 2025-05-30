#![feature(crate_local_distributed_slice)]

#[distributed_slice(crate)]
const MEOWS: [&str; _];

distributed_slice_element!(MEOWS, "mrow");

const THREE_MEOWS: [&str; 3] = ["mew", "prrr", "meow"];
distributed_slice_elements!(MEOWS, {
//~^ ERROR `distributed_slice_elements!()` only accepts a path or array literal
    THREE_MEOWS
});

fn main() {
    println!("{MEOWS:?}");
}
