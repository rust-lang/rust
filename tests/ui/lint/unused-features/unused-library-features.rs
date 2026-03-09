#![crate_type = "lib"]
#![deny(unused_features)]

// Unused library features
#![feature(step_trait)]
//~^ ERROR feature `step_trait` is declared but not used
#![feature(is_sorted)]
//~^ WARN the feature `is_sorted` has been stable since 1.82.0 and no longer requires an attribute to enable

// Enabled via cfg_attr, unused
#![cfg_attr(all(), feature(slice_ptr_get))]
//~^ ERROR feature `slice_ptr_get` is declared but not used
