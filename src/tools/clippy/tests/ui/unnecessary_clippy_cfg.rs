//@no-rustfix

#![allow(clippy::duplicated_attributes)]
#![warn(clippy::unnecessary_clippy_cfg)]
#![cfg_attr(clippy, deny(clippy::non_minimal_cfg))]
//~^ unnecessary_clippy_cfg
#![cfg_attr(clippy, deny(dead_code, clippy::non_minimal_cfg))]
//~^ unnecessary_clippy_cfg
#![cfg_attr(clippy, deny(dead_code, clippy::non_minimal_cfg))]
//~^ unnecessary_clippy_cfg
#![cfg_attr(clippy, deny(clippy::non_minimal_cfg))]
//~^ unnecessary_clippy_cfg

#[cfg_attr(clippy, deny(clippy::non_minimal_cfg))]
//~^ unnecessary_clippy_cfg
#[cfg_attr(clippy, deny(dead_code, clippy::non_minimal_cfg))]
//~^ unnecessary_clippy_cfg
#[cfg_attr(clippy, deny(dead_code, clippy::non_minimal_cfg))]
//~^ unnecessary_clippy_cfg
#[cfg_attr(clippy, deny(clippy::non_minimal_cfg))]
//~^ unnecessary_clippy_cfg

pub struct Bar;

fn main() {}
