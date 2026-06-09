//@ no-prefer-dynamic
#![crate_type = "rlib"]

#[link(name = "dummy", kind = "dylib")]
extern "C" {
    pub fn dylib_func2(x: i32) -> i32;
    pub static dylib_global2: i32;
}

#[link(name = "dummy", kind = "static")]
extern "C" {
    pub fn static_func2(x: i32) -> i32;
    pub static static_global2: i32;
}
