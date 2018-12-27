// Test that settingt the featute gate while using its functionality doesn't error.

// compile-pass

#![cfg_attr(all(), feature(cfg_attr_multi), crate_type="bin")]

fn main() {}
