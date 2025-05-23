#![feature(c_variadic)]
#![allow(anonymous_parameters)]

fn main() {}

fn f1_1(x: isize, ...) {}
//~^ ERROR only foreign, `unsafe extern "C"`, or `unsafe extern "C-unwind"` functions may have a C-variadic arg

fn f1_2(...) {}
//~^ ERROR only foreign, `unsafe extern "C"`, or `unsafe extern "C-unwind"` functions may have a C-variadic arg

extern "C" fn f2_1(x: isize, ...) {}
//~^ ERROR only foreign, `unsafe extern "C"`, or `unsafe extern "C-unwind"` functions may have a C-variadic arg

extern "C" fn f2_2(...) {}
//~^ ERROR only foreign, `unsafe extern "C"`, or `unsafe extern "C-unwind"` functions may have a C-variadic arg

extern "C" fn f2_3(..., x: isize) {}
//~^ ERROR only foreign, `unsafe extern "C"`, or `unsafe extern "C-unwind"` functions may have a C-variadic arg
//~| ERROR `...` must be the last argument of a C-variadic function

extern "C" fn f3_1(x: isize, ...) {}
//~^ ERROR only foreign, `unsafe extern "C"`, or `unsafe extern "C-unwind"` functions may have a C-variadic arg

extern "C" fn f3_2(...) {}
//~^ ERROR only foreign, `unsafe extern "C"`, or `unsafe extern "C-unwind"` functions may have a C-variadic arg

extern "C" fn f3_3(..., x: isize) {}
//~^ ERROR only foreign, `unsafe extern "C"`, or `unsafe extern "C-unwind"` functions may have a C-variadic arg
//~| ERROR `...` must be the last argument of a C-variadic function

const unsafe extern "C" fn f4_1(x: isize, ...) {}
//~^ ERROR functions cannot be both `const` and C-variadic

const extern "C" fn f4_2(x: isize, ...) {}
//~^ ERROR functions cannot be both `const` and C-variadic
//~| ERROR only foreign, `unsafe extern "C"`, or `unsafe extern "C-unwind"` functions may have a C-variadic arg

const extern "C" fn f4_3(..., x: isize, ...) {}
//~^ ERROR functions cannot be both `const` and C-variadic
//~| ERROR only foreign, `unsafe extern "C"`, or `unsafe extern "C-unwind"` functions may have a C-variadic arg
//~| ERROR `...` must be the last argument of a C-variadic function

extern "C" {
    fn e_f2(..., x: isize);
    //~^ ERROR `...` must be the last argument of a C-variadic function
}

struct X;

impl X {
    fn i_f1(x: isize, ...) {}
    //~^ ERROR only foreign, `unsafe extern "C"`, or `unsafe extern "C-unwind"` functions may have a C-variadic arg
    fn i_f2(...) {}
    //~^ ERROR only foreign, `unsafe extern "C"`, or `unsafe extern "C-unwind"` functions may have a C-variadic arg
    fn i_f3(..., x: isize, ...) {}
    //~^ ERROR only foreign, `unsafe extern "C"`, or `unsafe extern "C-unwind"` functions may have a C-variadic arg
    //~| ERROR `...` must be the last argument of a C-variadic function
    fn i_f4(..., x: isize, ...) {}
    //~^ ERROR only foreign, `unsafe extern "C"`, or `unsafe extern "C-unwind"` functions may have a C-variadic arg
    //~| ERROR `...` must be the last argument of a C-variadic function
    const fn i_f5(x: isize, ...) {}
    //~^ ERROR only foreign, `unsafe extern "C"`, or `unsafe extern "C-unwind"` functions may have a C-variadic arg
    //~| ERROR functions cannot be both `const` and C-variadic
}

trait T {
    fn t_f1(x: isize, ...) {}
    //~^ ERROR only foreign, `unsafe extern "C"`, or `unsafe extern "C-unwind"` functions may have a C-variadic arg
    fn t_f2(x: isize, ...);
    //~^ ERROR only foreign, `unsafe extern "C"`, or `unsafe extern "C-unwind"` functions may have a C-variadic arg
    fn t_f3(...) {}
    //~^ ERROR only foreign, `unsafe extern "C"`, or `unsafe extern "C-unwind"` functions may have a C-variadic arg
    fn t_f4(...);
    //~^ ERROR only foreign, `unsafe extern "C"`, or `unsafe extern "C-unwind"` functions may have a C-variadic arg
    fn t_f5(..., x: isize) {}
    //~^ ERROR only foreign, `unsafe extern "C"`, or `unsafe extern "C-unwind"` functions may have a C-variadic arg
    //~| ERROR `...` must be the last argument of a C-variadic function
    fn t_f6(..., x: isize);
    //~^ ERROR only foreign, `unsafe extern "C"`, or `unsafe extern "C-unwind"` functions may have a C-variadic arg
    //~| ERROR `...` must be the last argument of a C-variadic function
}
