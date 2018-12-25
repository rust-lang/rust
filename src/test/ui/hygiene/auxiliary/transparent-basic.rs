#![feature(decl_macro, rustc_attrs)]

#[rustc_transparent_macro]
pub macro dollar_crate() {
    let s = $crate::S;
}
