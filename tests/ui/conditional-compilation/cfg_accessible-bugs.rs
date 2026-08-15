// This test is a collection of test that should pass.
//
//@ check-fail

#![feature(cfg_accessible)]
#![feature(trait_alias)]

trait TraitAlias = std::fmt::Debug + Send;

// FIXME: Currently shows "cannot determine" but should be `false`
#[cfg_accessible(unresolved)] //~ ERROR cannot determine
const C: bool = true;

// FIXME: Currently shows "not sure" but should be `false`
#[cfg_accessible(TraitAlias::unresolved)] //~ ERROR not sure whether the path is accessible or not
const D: bool = true;

fn main() {}
