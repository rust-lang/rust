//@ check-pass
//@ compile-flags: --check-cfg=cfg()

/// uniz is nor a builtin nor pass as arguments so is unexpected
#[cfg(uniz)]
//~^ WARNING unexpected `cfg` condition name
pub struct Bar;
