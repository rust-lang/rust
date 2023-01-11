#![allow(non_camel_case_types)]

struct bool;

fn foo(_: bool) {}

fn main() {
    foo(true);
    //~^ ERROR mismatched types [E0308]
}
