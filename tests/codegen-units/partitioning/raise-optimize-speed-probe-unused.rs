//@ incremental
//@ compile-flags: -Copt-level=0 -Zraise-optimize-speed=modules,probe=added-unused

#![crate_type = "lib"]
// The function that the compiler adds is not reported as dead code.
#![deny(dead_code)]

// Under `-Z raise-optimize-speed=modules,probe=added-unused`, the function that the compiler adds
// to the crate is collected though nothing refers to it, is assigned to the `speed` CGU of the
// crate root, and stays external there, so that code is emitted for it. No static is added, and
// the crate's own items stay where they are.

//~ MONO_ITEM fn _::_probe @@ raise_optimize_speed_probe_unused.speed[External]

//~ MONO_ITEM fn plain @@ raise_optimize_speed_probe_unused[External]
pub fn plain() {}
