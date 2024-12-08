// Check what happens when `function` is used to define a function, instead of `fn`
//@ edition:2021

#![allow(dead_code)]

function foo() {}
//~^ ERROR expected one of `!` or `::`, found `foo`
//~^^ HELP write `fn` instead of `function` to declare a function

fn main() {}
