#![warn(unused)]
#![allow(dead_code)]
#![deny(non_snake_case)]

mod Impl {}
//~^ ERROR module `Impl` should have a snake case name

fn While() {}
//~^ ERROR function `While` should have a snake case name

fn main() {
    let Mod: usize = 0;
    //~^ ERROR variable `Mod` should have a snake case name
    //~^^ WARN unused variable: `Mod`

    let Super: usize = 0;
    //~^ ERROR variable `Super` should have a snake case name
    //~^^ WARN unused variable: `Super`
}
