//@ known-bug: #127676
//@ edition:2018

#![feature(dyn_star,const_async_blocks)]

static S: dyn* Send + Sync = async { 42 };

pub fn main() {}
