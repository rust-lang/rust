// Check what happens when `def` is used to define a function, instead of `fn`
//@ edition:2021

#![allow(dead_code)]

def foo() {}
//~^ ERROR expected one of `!` or `::`, found `foo`
//~^^ HELP write `fn` instead of `def` to declare a function

fn main() {}
