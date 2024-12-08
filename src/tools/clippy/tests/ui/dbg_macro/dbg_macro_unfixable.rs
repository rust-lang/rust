//@no-rustfix
#![warn(clippy::dbg_macro)]

#[path = "auxiliary/submodule.rs"]
mod submodule;

fn main() {
    dbg!(dbg!(dbg!(42)));
    //~^ ERROR: the `dbg!` macro is intended as a debugging tool
    dbg!(1, 2, dbg!(3, 4));
    //~^ ERROR: the `dbg!` macro is intended as a debugging tool
}
