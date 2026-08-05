//@ check-fail
//@ compile-flags:--cfg foo --check-cfg=cfg(foo)
//@ reference: cfg.cfg_attr.attr-restriction

#![cfg_attr(foo, crate_type="bin")]
//~^ERROR `crate_type` attribute is forbidden within
//~|ERROR `crate_type` attribute is forbidden within
#![cfg_attr(foo, crate_name="bar")]
//~^ERROR `crate_name` attribute is forbidden within
//~|ERROR `crate_name` attribute is forbidden within

fn main() {}
