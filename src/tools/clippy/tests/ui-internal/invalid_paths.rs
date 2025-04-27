#![deny(clippy::invalid_paths)]
#![allow(clippy::missing_clippy_version_attribute, clippy::unnecessary_def_path)]

mod paths {
    // Good path
    pub const ANY_TRAIT: [&str; 3] = ["std", "any", "Any"];

    // Path to method on inherent impl of a primitive type
    pub const F32_EPSILON: [&str; 4] = ["core", "f32", "<impl f32>", "EPSILON"];

    // Path to method on inherent impl
    pub const ARC_PTR_EQ: [&str; 4] = ["alloc", "sync", "Arc", "ptr_eq"];

    // Path with empty segment
    pub const TRANSMUTE: [&str; 4] = ["core", "intrinsics", "", "transmute"];
    //~^ invalid_paths

    // Path with bad crate
    pub const BAD_CRATE_PATH: [&str; 2] = ["bad", "path"];
    //~^ invalid_paths

    // Path with bad module
    pub const BAD_MOD_PATH: [&str; 2] = ["std", "xxx"];
    //~^ invalid_paths

    // Path to method on an enum inherent impl
    pub const OPTION_IS_SOME: [&str; 4] = ["core", "option", "Option", "is_some"];
}

fn main() {}
