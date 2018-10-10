// Test that settingt the featute gate while using its functionality doesn't error.
// Specifically, if there's a cfg-attr *before* the feature gate.

// compile-pass

#![cfg_attr(all(),)]
#![cfg_attr(all(), feature(cfg_attr_multi), crate_type="bin")]

fn main() {}
