#![feature(decl_macro, rustc_attrs)]

#[rustc_macro_transparency = "transparent"]
macro transparent() {
    struct Transparent;
    let transparent = 0;
}
#[rustc_macro_transparency = "semitransparent"]
macro semiopaque() {
    struct SemiOpaque;
    let semiopaque = 0;
}
#[rustc_macro_transparency = "opaque"]
macro opaque() {
    struct Opaque;
    let opaque = 0;
}

fn main() {
    transparent!();
    semiopaque!();
    opaque!();

    Transparent; // OK
    SemiOpaque; // OK
    Opaque; //~ ERROR cannot find value `Opaque` in this scope

    transparent; // OK
    semiopaque; //~ ERROR expected value, found macro `semiopaque`
    opaque; //~ ERROR expected value, found macro `opaque`
}
