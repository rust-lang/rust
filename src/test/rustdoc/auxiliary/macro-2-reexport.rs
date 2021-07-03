#![crate_name = "macro_2_reexport"]
#![feature(decl_macro)]

pub macro addr_of($place:expr) {
    &raw const $place
}
