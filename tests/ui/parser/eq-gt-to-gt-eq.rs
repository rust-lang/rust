//@ run-rustfix
// Check that we try to correct `=>` to `>=` in conditions.
#![allow(unused)]

fn main() {
    let a = 0;
    let b = 1;
    if a => b {} //~ERROR
}

fn foo() {
    let a = 0;
    if a => 1 {} //~ERROR
}

fn a() {
    let a = 0;
    if 1 => a {} //~ERROR
}

fn bar() {
    let a = 0;
    let b = 1;
    if a => b && a != b {} //~ERROR
}

fn qux() {
    let a = 0;
    let b = 1;
    if a != b && a => b {} //~ERROR
}

fn baz() {
    let a = 0;
    let b = 1;
    let _ = a => b; //~ERROR
}

fn b() {
    let a = 0;
    let b = 1;
    match a => b { //~ERROR
        _ => todo!(),
    }
}
