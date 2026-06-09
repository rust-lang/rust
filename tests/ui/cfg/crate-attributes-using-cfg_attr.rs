//@ check-fail
//@ compile-flags:--cfg foo --check-cfg=cfg(foo)
//@ reference: cfg.cfg_attr.attr-restriction

#![cfg_attr(foo, crate_type="bin")]
//~^ERROR `crate_type` within
//~|ERROR `crate_type` within
#![cfg_attr(foo, crate_name="bar")]
//~^ERROR `crate_name` within
//~|ERROR `crate_name` within

fn main() {}
