#![feature(link_cfg)]
#![crate_type = "rlib"]

#[link(name = "one", kind = "static", cfg(use_one))]
#[link(name = "two", kind = "static", cfg(use_two))]
extern "C" {
    fn native_value() -> i32;
}

#[link(name = "shared", kind = "static", cfg(shared_x))]
#[link(name = "shared", kind = "static", cfg(shared_y))]
extern "C" {}

#[link(name = "dup_a", kind = "static", cfg(dup_a))]
#[link(name = "dup_b", kind = "static", cfg(dup_b))]
extern "C" {}

pub fn value() -> i32 {
    unsafe { native_value() }
}
