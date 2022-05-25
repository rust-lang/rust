// This test is a collection of test that should pass.
//
// check-fail

#![feature(cfg_accessible)]
#![feature(trait_alias)]

enum Enum {
    Existing { existing: u8 },
}

trait TraitAlias = std::fmt::Debug + Send;

// FIXME: Currently returns `false` but should be "not sure"
#[cfg_accessible(Enum::Existing::existing)]
const A: bool = true;

// FIXME: Currently returns `false` but should be "not sure"
#[cfg_accessible(Enum::Existing::unresolved)]
const B: bool = true;

// FIXME: Currently shows "cannot determine" but should be `false`
#[cfg_accessible(unresolved)] //~ ERROR cannot determine
const C: bool = true;

// FIXME: Currently shows "not sure" but should be `false`
#[cfg_accessible(TraitAlias::unresolved)] //~ ERROR not sure whether the path is accessible or not
const D: bool = true;

fn main() {}
