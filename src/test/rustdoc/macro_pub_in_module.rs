//! See issue #74355
#![feature(decl_macro, no_core, rustc_attrs)]
#![crate_name = "krate"]
#![no_core]

pub mod inner {
    // @has krate/inner/macro.my_macro.html
    pub macro my_macro() {}

    // @has krate/inner/macro.test.html
    #[rustc_builtin_macro]
    pub macro test($item:item) {}

    // @has krate/inner/macro.Clone.html
    #[rustc_builtin_macro]
    pub macro Clone($item:item) {}
}
