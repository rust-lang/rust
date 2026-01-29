//@ compile-flags: --crate-type=lib

#[cfg(target_modifier_fixed_x18)]
//~^ ERROR: `cfg(target_modifier_fixed_x18)` is experimental and subject to change
fn branch_protection() {}
