//@ run-rustfix
use std::collections::BTreeMap;

fn main() {
    println!("{}", std::mem:size_of::<BTreeMap<u32, u32>>());
    //~^ ERROR path separator must be a double colon
}
