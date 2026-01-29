//@ compile-flags: --crate-type=lib
//@ revisions: feat no_feat
//@[feat] check-pass
//@[no_feat] check-fail

#![allow(unexpected_cfgs)]
#![cfg_attr(feat, feature(cfg_unstable_target_modifier))]

#[cfg(target_modifier_fixed_x18)]
//[no_feat]~^ ERROR: `cfg(target_modifier_fixed_x18)` is experimental and subject to change
fn fixed_x18() {}

#[cfg(target_modifier_indirect_branch_cs_prefix)]
//[no_feat]~^ ERROR: `cfg(target_modifier_indirect_branch_cs_prefix)` is experimental and subject to change
fn indirect_branch_cs_prefix() {}

#[cfg(target_modifier_reg_struct_return)]
//[no_feat]~^ ERROR: `cfg(target_modifier_reg_struct_return)` is experimental and subject to change
fn reg_struct_return() {}

#[cfg(target_modifier_regparm="0")]
//[no_feat]~^ ERROR: `cfg(target_modifier_regparm)` is experimental and subject to change
fn regparm() {}

#[cfg(target_modifier_retpoline)]
//[no_feat]~^ ERROR: `cfg(target_modifier_retpoline)` is experimental and subject to change
fn retpoline() {}

#[cfg(target_modifier_retpoline_external_thunk)]
//[no_feat]~^ ERROR: `cfg(target_modifier_retpoline_external_thunk)` is experimental and subject to change
fn retpoline_external_thunk() {}
