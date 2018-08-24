#![feature(type_ascription)]

fn main() {
    f()  :
    f(); //~ ERROR expected type, found function
}

fn f() {}
