// Test non-power-of-two alignment.

#![feature(rustc_attrs)]

#[path = "../../utils/mod.no_std.rs"]
mod utils;

fn main() {
    unsafe {
        utils::miri_alloc(1, 3);
        //~^ERROR: creating allocation with non-power-of-two alignment
    }
}
