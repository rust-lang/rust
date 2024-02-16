//@no-rustfix

#![warn(clippy::unnecessary_clippy_cfg)]
#![cfg_attr(clippy, deny(clippy::non_minimal_cfg))]
//~^ ERROR: no need to put clippy lints behind a `clippy` cfg
#![cfg_attr(clippy, deny(dead_code, clippy::non_minimal_cfg))]
//~^ ERROR: no need to put clippy lints behind a `clippy` cfg
#![cfg_attr(clippy, deny(dead_code, clippy::non_minimal_cfg, clippy::maybe_misused_cfg))]
//~^ ERROR: no need to put clippy lints behind a `clippy` cfg
#![cfg_attr(clippy, deny(clippy::non_minimal_cfg, clippy::maybe_misused_cfg))]
//~^ ERROR: no need to put clippy lints behind a `clippy` cfg

#[cfg_attr(clippy, deny(clippy::non_minimal_cfg))]
//~^ ERROR: no need to put clippy lints behind a `clippy` cfg
#[cfg_attr(clippy, deny(dead_code, clippy::non_minimal_cfg))]
//~^ ERROR: no need to put clippy lints behind a `clippy` cfg
#[cfg_attr(clippy, deny(dead_code, clippy::non_minimal_cfg, clippy::maybe_misused_cfg))]
//~^ ERROR: no need to put clippy lints behind a `clippy` cfg
#[cfg_attr(clippy, deny(clippy::non_minimal_cfg, clippy::maybe_misused_cfg))]
//~^ ERROR: no need to put clippy lints behind a `clippy` cfg
pub struct Bar;

fn main() {}
