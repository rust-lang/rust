// build-pass (FIXME(62277): could be check-pass?)

#![allow(unused_variables)]
#![allow(dead_code)]
#![deny(unreachable_code)]

fn foo() {
    let x = false || (return);
    println!("I am not dead.");
}

fn main() { }
