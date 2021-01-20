#![feature(link_cfg)]
#![crate_type = "rlib"]

#[link(name = "return1", cfg(foo))]
#[link(name = "return2", cfg(bar))]
extern {
    pub fn my_function() -> i32;
}
