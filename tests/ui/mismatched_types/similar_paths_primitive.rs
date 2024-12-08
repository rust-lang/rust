#![allow(non_camel_case_types)]

struct bool;
struct str;

fn foo(_: bool) {}
fn bar(_: &str) {}

fn main() {
    foo(true);
    //~^ ERROR mismatched types [E0308]
    bar("hello");
    //~^ ERROR mismatched types [E0308]
}
