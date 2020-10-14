// aux-build:macro_pub_in_module.rs
// edition:2018
// build-aux-docs
// @has external_crate/some_module/macro.external_macro.html

//! See issue #74355
#![feature(decl_macro, no_core, rustc_attrs)]
#![crate_name = "krate"]
#![no_core]

extern crate external_crate;

pub mod inner {
    // @has krate/inner/macro.raw_const.html
    pub macro raw_const() {}

    // @has krate/inner/macro.test.html
    #[rustc_builtin_macro]
    pub macro test($item:item) {}

    // @has krate/inner/macro.Clone.html
    #[rustc_builtin_macro]
    pub macro Clone($item:item) {}

    // Make sure the logic is not affected by a re-export.
    mod private {
        pub macro m() {}
    }
    // @has krate/inner/macro.renamed.html
    pub use private::m as renamed;

    // @has krate/inner/macro.external_macro.html
    pub use ::external_crate::some_module::external_macro;
}

// Namespaces: Make sure the logic does not mix up a function name with a module name…
fn both_fn_and_mod() {
    pub macro m() {}
}
pub mod both_fn_and_mod {
    // @!has krate/both_fn_and_mod/macro.m.html
}

const __: () = {
    pub macro m() {}
};
pub mod __ {
    // @!has krate/__/macro.m.html
}
