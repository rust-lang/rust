//@ aux-build:macro_pub_in_module.rs
//@ edition:2018
//@ build-aux-docs

//! See issue #74355
#![feature(decl_macro, no_core, rustc_attrs)]
#![crate_name = "krate"]
#![no_core]

//@ has external_crate/some_module/macro.external_macro.html
//@ !has external_crate/macro.external_macro.html
extern crate external_crate;

pub mod inner {
    //@ has krate/inner/macro.raw_const.html
    //@ !has krate/macro.raw_const.html
    pub macro raw_const() {}

    //@ has krate/inner/attr.test.html
    //@ !has krate/macro.test.html
    //@ !has krate/inner/macro.test.html
    //@ !has krate/attr.test.html
    #[rustc_builtin_macro]
    pub macro test($item:item) {}

    //@ has krate/inner/derive.Clone.html
    //@ !has krate/inner/macro.Clone.html
    //@ !has krate/macro.Clone.html
    //@ !has krate/derive.Clone.html
    #[rustc_builtin_macro]
    pub macro Clone($item:item) {}

    // Make sure the logic is not affected by re-exports.
    mod unrenamed {
        //@ !has krate/macro.unrenamed.html
        #[rustc_macro_transparency = "semitransparent"]
        pub macro unrenamed() {}
    }
    //@ has krate/inner/macro.unrenamed.html
    pub use unrenamed::unrenamed;

    mod private {
        //@ !has krate/macro.m.html
        pub macro m() {}
    }
    //@ has krate/inner/macro.renamed.html
    //@ !has krate/macro.renamed.html
    pub use private::m as renamed;

    mod private2 {
        //@ !has krate/macro.m2.html
        pub macro m2() {}
    }
    use private2 as renamed_mod;
    //@ has krate/inner/macro.m2.html
    pub use renamed_mod::m2;

    //@ has krate/inner/macro.external_macro.html
    //@ !has krate/macro.external_macro.html
    pub use ::external_crate::some_module::external_macro;
}

// Namespaces: Make sure the logic does not mix up a function name with a module name…
fn both_fn_and_mod() {
    //@ !has krate/macro.in_both_fn_and_mod.html
    pub macro in_both_fn_and_mod() {}
}
pub mod both_fn_and_mod {
    //@ !has krate/both_fn_and_mod/macro.in_both_fn_and_mod.html
}

const __: () = {
    //@ !has krate/macro.in_both_const_and_mod.html
    pub macro in_both_const_and_mod() {}
};
pub mod __ {
    //@ !has krate/__/macro.in_both_const_and_mod.html
}

enum Enum {
    Crazy = {
        //@ !has krate/macro.this_is_getting_weird.html;
        pub macro this_is_getting_weird() {}
        42
    },
}
