#![allow(warnings)]
#![deny(unreachable_code)]

enum Void { }

fn bar() -> Void {
    unreachable!()
}

fn foo() {
    match bar() { }
    let x = 2; //~ ERROR unreachable
}

fn main() {}
