#![feature(c_variadic)]

fn main() {}

fn f1(x: isize, ...) {}
//~^ ERROR: only foreign or `unsafe extern "C" functions may be C-variadic

extern "C" fn f2(x: isize, ...) {}
//~^ ERROR: only foreign or `unsafe extern "C" functions may be C-variadic

extern fn f3(x: isize, ...) {}
//~^ ERROR: only foreign or `unsafe extern "C" functions may be C-variadic

struct X;

impl X {
    fn f4(x: isize, ...) {}
    //~^ ERROR: only foreign or `unsafe extern "C" functions may be C-variadic
}

trait T {
    fn f5(x: isize, ...) {}
    //~^ ERROR: only foreign or `unsafe extern "C" functions may be C-variadic
    fn f6(x: isize, ...);
    //~^ ERROR: only foreign or `unsafe extern "C" functions may be C-variadic
}
