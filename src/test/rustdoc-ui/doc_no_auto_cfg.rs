// edition: 2021
#![deny(warnings)]
#![doc(no_auto_cfg)] //~ ERROR
//~^ ERROR
#![doc(auto_cfg, no_auto_cfg)] //~ ERROR
#![doc(no_auto_cfg(1))] //~ ERROR
#![doc(no_auto_cfg = 1)] //~ ERROR
#![doc(auto_cfg(1))] //~ ERROR
#![doc(auto_cfg = 1)] //~ ERROR

#[doc(auto_cfg)] //~ ERROR
//~^ WARN
#[doc(no_auto_cfg)] //~ ERROR
//~^ WARN
pub struct Bar;
