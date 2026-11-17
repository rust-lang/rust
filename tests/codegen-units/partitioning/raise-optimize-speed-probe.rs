//@ incremental
//@ compile-flags: -Copt-level=0 -Zraise-optimize-speed=modules,probe=added

#![crate_type = "lib"]

// Under `-Z raise-optimize-speed=modules,probe=added`, the function that the compiler adds to
// the crate and the static that keeps it alive are both assigned to the `speed` CGU of the crate
// root, and the crate's own items stay where they are.

//~ MONO_ITEM fn _::probe @@ raise_optimize_speed_probe.speed[External]
//~ MONO_ITEM static _::PROBE @@ raise_optimize_speed_probe.speed[External]

//~ MONO_ITEM fn plain @@ raise_optimize_speed_probe[External]
pub fn plain() {}
