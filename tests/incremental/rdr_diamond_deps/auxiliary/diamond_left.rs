#![crate_name = "diamond_left"]
#![crate_type = "rlib"]

extern crate diamond_base;

pub fn left_value() -> u32 {
    diamond_base::base_value() + 1
}
