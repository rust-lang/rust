#![feature(never_type)]
#![feature(never_type_fallback)]
#![allow(dead_code)]

fn foo() {
    { if true { return } else { return } }.test();
    //~^ ERROR no method named `test` found for type `!`
}

fn bar() {
    { if true { Default::default() } else { return } }.test();
    //~^ ERROR type annotations needed
}

fn baz() {
    let a = return;
    { if true { return } else { a } }.test();
    //~^ ERROR type annotations needed
}

fn qux() {
    let a: ! = return;
    { if true { return } else { a } }.test();
    //~^ ERROR no method named `test` found for type `!`
}

fn main() {}
