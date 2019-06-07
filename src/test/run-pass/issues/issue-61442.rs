#![feature(generators)]

fn foo() {
    let _x = static || {
        let mut s = String::new();
        s += { yield; "" };
    };

    let _y = static || {
        let x = &mut 0;
        *{ yield; x } += match String::new() { _ => 0 };
    };

    // Please don't ever actually write something like this
    let _z = static || {
        let x = &mut 0;
        *{
            let inner = &mut 1;
            *{ yield (); inner } += match String::new() { _ => 1};
            yield;
            x
        } += match String::new() { _ => 2 };
    };
}

fn main() {
    foo()
}
