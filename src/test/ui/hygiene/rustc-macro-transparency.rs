#![feature(decl_macro, rustc_attrs)]

#[rustc_macro_transparency = "transparent"]
macro transparent() {
    struct Transparent;
    let transparent = 0;
}
#[rustc_macro_transparency = "semitransparent"]
macro semitransparent() {
    struct SemiTransparent;
    let semitransparent = 0;
}
#[rustc_macro_transparency = "opaque"]
macro opaque() {
    struct Opaque;
    let opaque = 0;
}

fn main() {
    transparent!();
    semitransparent!();
    opaque!();

    Transparent; // OK
    SemiTransparent; // OK
    Opaque; //~ ERROR cannot find value `Opaque` in this scope

    transparent; // OK
    semitransparent; //~ ERROR cannot find value `semitransparent` in this scope
    opaque; //~ ERROR cannot find value `opaque` in this scope
}
