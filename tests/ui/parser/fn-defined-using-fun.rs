// Check what happens when `fun` is used to define a function, instead of `fn`
//@ edition:2021

#![allow(dead_code)]

fun foo() {}
//~^ ERROR expected one of `!` or `::`, found `foo`
//~^^ HELP write `fn` instead of `fun` to declare a function

fn main() {}
