//@ check-pass
//@ compile-flags: --check-cfg=cfg() -Z unstable-options

/// uniz is nor a builtin nor pass as arguments so is unexpected
#[cfg(uniz)]
//~^ WARNING unexpected `cfg` condition name
pub struct Bar;
