#![feature(decl_macro, rustc_attrs)]

#[rustc_macro_transparency = "transparent"]
pub macro dollar_crate() {
    let s = $crate::S;
}
