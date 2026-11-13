//@no-rustfix: overlapping suggestions
//@error-in-other-file:
#![warn(clippy::dbg_macro)]

#[path = "auxiliary/submodule.rs"]
mod submodule;

fn main() {
    dbg!(dbg!(dbg!(42)));
    //~^ dbg_macro
    //~| dbg_macro
    //~| dbg_macro

    dbg!(1, 2, dbg!(3, 4));
    //~^ dbg_macro
    //~| dbg_macro
}
