#![crate_name = "diamond_right"]
#![crate_type = "rlib"]

extern crate diamond_base;

pub fn right_value() -> u32 {
    diamond_base::base_value() + 2
}
