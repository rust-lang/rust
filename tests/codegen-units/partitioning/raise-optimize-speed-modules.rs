//@ incremental
//@ compile-flags: -Copt-level=0 -Zraise-optimize-speed=modules

#![crate_type = "lib"]

// Under `-Z raise-optimize-speed=modules`, a function marked `#[optimize(speed)]` never shares
// a CGU with a function that lacks the mark: it is assigned to the `speed` CGU that stands
// beside the CGU of its module. A closure in a marked function goes where the function goes.

//~ MONO_ITEM fn plain @@ raise_optimize_speed_modules[External]
pub fn plain() {}

//~ MONO_ITEM fn marked @@ raise_optimize_speed_modules.speed[External]
#[optimize(speed)]
pub fn marked() {}

pub mod mod1 {
    //~ MONO_ITEM fn mod1::plain @@ raise_optimize_speed_modules-mod1[External]
    pub fn plain() {}

    //~ MONO_ITEM fn mod1::marked @@ raise_optimize_speed_modules-mod1.speed[External]
    //~ MONO_ITEM fn mod1::marked::{closure#0} @@ raise_optimize_speed_modules-mod1.speed[Internal]
    #[optimize(speed)]
    pub fn marked() -> u32 {
        let one = || 1;
        one()
    }
}
