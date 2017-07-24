#![feature(never_type)]
#![allow(unreachable_code)]
#![allow(unused_variables)]

enum Void {}

fn f(v: Void) -> ! {
    match v {}
}

fn main() {
    let v: Void = unsafe {
        std::mem::transmute::<(), Void>(()) //~ ERROR entered unreachable code
    };
    f(v);
}
