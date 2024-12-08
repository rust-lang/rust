//@ check-pass

#![feature(dyn_star)]
//~^ WARN the feature `dyn_star` is incomplete

const C: dyn* Send + Sync = &();

static S: dyn* Send + Sync = &();

fn main() {}
