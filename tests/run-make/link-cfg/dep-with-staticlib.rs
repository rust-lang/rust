#![feature(link_cfg)]
#![crate_type = "rlib"]

#[link(name = "return1", cfg(foo))]
#[link(name = "return3", kind = "static", cfg(bar))]
extern "C" {
    pub fn my_function() -> i32;
}
