//! See issue #74355
#![crate_name = "krate"]
#![feature(decl_macro)]

// @has krate/some_module/macro.my_macro.html
pub mod some_module {
    //
    pub macro my_macro() {}
}
