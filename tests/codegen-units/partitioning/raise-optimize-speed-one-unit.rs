//@ incremental
//@ compile-flags: -Copt-level=0 -Ccodegen-units=1 -Zraise-optimize-speed=modules

#![crate_type = "lib"]

// `-C codegen-units=1` bounds the CGUs of each kind. Under `-Z raise-optimize-speed=modules`,
// a function marked `#[optimize(speed)]` still gets a CGU that holds no function that lacks the
// mark, beside the one CGU for the rest of the crate.

//~ MONO_ITEM fn plain @@ raise_optimize_speed_one_unit[External]
pub fn plain() {}

//~ MONO_ITEM fn marked @@ raise_optimize_speed_one_unit.speed[External]
#[optimize(speed)]
pub fn marked() {}
