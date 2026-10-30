#![warn(clippy::ifs_as_logical_ops)]
//@no-rustfix
fn unfixable_example(x: bool, y: bool, z: bool) -> bool {
    if x { if y { z } else { false } } else { false }
    //~^ ifs_as_logical_ops
    //~| ifs_as_logical_ops
}
