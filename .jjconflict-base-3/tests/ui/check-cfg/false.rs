//! Check that `cfg(false)` is suggested instead of cfg(FALSE)
//
//@ check-pass
//@ no-auto-check-cfg
//@ compile-flags: --check-cfg=cfg()

#[cfg(FALSE)]
//~^ WARNING unexpected `cfg` condition name: `FALSE`
//~| HELP: to expect this configuration use
//~| HELP: you may have meant to use `false` (notice the capitalization).
pub fn a() {}

#[cfg(False)]
//~^ WARNING unexpected `cfg` condition name: `False`
//~| HELP: to expect this configuration use
//~| HELP: you may have meant to use `false` (notice the capitalization).
pub fn b() {}

#[cfg(r#false)]
//~^ WARNING unexpected `cfg` condition name: `r#false`
//~| HELP: to expect this configuration use
// No capitalization help for r#false
pub fn c() {}

#[cfg(r#False)]
//~^ WARNING unexpected `cfg` condition name: `False`
//~| HELP: to expect this configuration use
// No capitalization help for r#False
pub fn d() {}

#[cfg(false)]
pub fn e() {}

#[cfg(TRUE)]
//~^ WARNING unexpected `cfg` condition name: `TRUE`
//~| HELP: to expect this configuration use
//~| HELP: you may have meant to use `true` (notice the capitalization).
pub fn f() {}

#[cfg(True)]
//~^ WARNING unexpected `cfg` condition name: `True`
//~| HELP: to expect this configuration use
//~| HELP: you may have meant to use `true` (notice the capitalization).
pub fn g() {}

#[cfg(r#true)]
//~^ WARNING unexpected `cfg` condition name: `r#true`
//~| HELP: to expect this configuration use
// No capitalization help for r#true
pub fn h() {}

#[cfg(r#True)]
//~^ WARNING unexpected `cfg` condition name: `True`
//~| HELP: to expect this configuration use
// No capitalization help for r#True
pub fn i() {}

#[cfg(true)]
pub fn j() {}

pub fn main() {}
