//@ edition:2018

#![feature(dyn_star, const_async_blocks)]
//~^ WARN the feature `dyn_star` is incomplete

static S: dyn* Send + Sync = async { 42 };
//~^ ERROR needs to have the same ABI as a pointer

pub fn main() {}
