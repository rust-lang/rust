// Tests that where a single file is referred to in multiple places, we don't
// crash.

#[cfg(all(foo))]
#[path = "closure.rs"]
pub mod imp;

#[cfg(all(bar))]
#[path = "closure.rs"]
pub mod imp;
