#![feature(decl_macro, rustc_attrs)]

#[rustc_macro_transparency = "transparent"]
macro transparent() {
    struct Transparent;
    let transparent = 0;
}
#[rustc_macro_transparency = "semiopaque"]
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
    semiopaque; //~ ERROR cannot find value `semiopaque` in this scope
    opaque; //~ ERROR cannot find value `opaque` in this scope
}
