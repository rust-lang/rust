//@ incremental
//@ compile-flags: -Copt-level=0 -Zraise-optimize-speed=one-module

#![crate_type = "lib"]

// Under `-Z raise-optimize-speed=one-module`, every function marked `#[optimize(speed)]` is
// assigned to one CGU made for such functions, whatever module it is defined in, and no
// function that lacks the mark is.

//~ MONO_ITEM fn plain @@ raise_optimize_speed_one_module[External]
pub fn plain() {}

//~ MONO_ITEM fn marked @@ raise_optimize_speed_one_module-optimize.speed[External]
#[optimize(speed)]
pub fn marked() {}

pub mod mod1 {
    //~ MONO_ITEM fn mod1::plain @@ raise_optimize_speed_one_module-mod1[External]
    pub fn plain() {}

    //~ MONO_ITEM fn mod1::marked @@ raise_optimize_speed_one_module-optimize.speed[External]
    #[optimize(speed)]
    pub fn marked() {}
}
