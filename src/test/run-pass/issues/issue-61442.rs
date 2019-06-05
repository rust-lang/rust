#![feature(generators)]

fn foo() {
    let _x = static || {
        let mut s = String::new();
        s += { yield; "" };
    };
}

fn main() {
    foo()
}
