#![feature(c_variadic)]
#![allow(anonymous_parameters)]

fn main() {}

fn f1_1(x: isize, ...) {}
//~^ ERROR defining functions with C-variadic arguments is only allowed for free functions with the "C" or "C-unwind" calling convention

fn f1_2(...) {}
//~^ ERROR defining functions with C-variadic arguments is only allowed for free functions with the "C" or "C-unwind" calling convention

extern "C" fn f2_1(x: isize, ...) {}
//~^ ERROR defining functions with C-variadic arguments is only allowed for free functions with the "C" or "C-unwind" calling convention

extern "C" fn f2_2(...) {}
//~^ ERROR defining functions with C-variadic arguments is only allowed for free functions with the "C" or "C-unwind" calling convention

extern "C" fn f2_3(..., x: isize) {}
//~^ ERROR defining functions with C-variadic arguments is only allowed for free functions with the "C" or "C-unwind" calling convention
//~| ERROR `...` must be the last argument of a C-variadic function

extern "C" fn f3_1(x: isize, ...) {}
//~^ ERROR defining functions with C-variadic arguments is only allowed for free functions with the "C" or "C-unwind" calling convention

extern "C" fn f3_2(...) {}
//~^ ERROR defining functions with C-variadic arguments is only allowed for free functions with the "C" or "C-unwind" calling convention

extern "C" fn f3_3(..., x: isize) {}
//~^ ERROR defining functions with C-variadic arguments is only allowed for free functions with the "C" or "C-unwind" calling convention
//~| ERROR `...` must be the last argument of a C-variadic function

const unsafe extern "C" fn f4_1(x: isize, ...) {}
//~^ ERROR functions cannot be both `const` and C-variadic
//~| ERROR destructor of `VaListImpl<'_>` cannot be evaluated at compile-time

const extern "C" fn f4_2(x: isize, ...) {}
//~^ ERROR functions cannot be both `const` and C-variadic
//~| ERROR defining functions with C-variadic arguments is only allowed for free functions with the "C" or "C-unwind" calling convention
//~| ERROR destructor of `VaListImpl<'_>` cannot be evaluated at compile-time

const extern "C" fn f4_3(..., x: isize, ...) {}
//~^ ERROR functions cannot be both `const` and C-variadic
//~| ERROR defining functions with C-variadic arguments is only allowed for free functions with the "C" or "C-unwind" calling convention
//~| ERROR `...` must be the last argument of a C-variadic function

extern "C" {
    fn e_f2(..., x: isize);
    //~^ ERROR `...` must be the last argument of a C-variadic function
}

struct X;

impl X {
    fn i_f1(x: isize, ...) {}
    //~^ ERROR defining functions with C-variadic arguments is only allowed for free functions with the "C" or "C-unwind" calling convention
    fn i_f2(...) {}
    //~^ ERROR defining functions with C-variadic arguments is only allowed for free functions with the "C" or "C-unwind" calling convention
    fn i_f3(..., x: isize, ...) {}
    //~^ ERROR defining functions with C-variadic arguments is only allowed for free functions with the "C" or "C-unwind" calling convention
    //~| ERROR `...` must be the last argument of a C-variadic function
    fn i_f4(..., x: isize, ...) {}
    //~^ ERROR defining functions with C-variadic arguments is only allowed for free functions with the "C" or "C-unwind" calling convention
    //~| ERROR `...` must be the last argument of a C-variadic function
    const fn i_f5(x: isize, ...) {}
    //~^ ERROR defining functions with C-variadic arguments is only allowed for free functions with the "C" or "C-unwind" calling convention
    //~| ERROR functions cannot be both `const` and C-variadic
    //~| ERROR destructor of `VaListImpl<'_>` cannot be evaluated at compile-time
}

trait T {
    fn t_f1(x: isize, ...) {}
    //~^ ERROR defining functions with C-variadic arguments is only allowed for free functions with the "C" or "C-unwind" calling convention
    fn t_f2(x: isize, ...);
    //~^ ERROR defining functions with C-variadic arguments is only allowed for free functions with the "C" or "C-unwind" calling convention
    fn t_f3(...) {}
    //~^ ERROR defining functions with C-variadic arguments is only allowed for free functions with the "C" or "C-unwind" calling convention
    fn t_f4(...);
    //~^ ERROR defining functions with C-variadic arguments is only allowed for free functions with the "C" or "C-unwind" calling convention
    fn t_f5(..., x: isize) {}
    //~^ ERROR defining functions with C-variadic arguments is only allowed for free functions with the "C" or "C-unwind" calling convention
    //~| ERROR `...` must be the last argument of a C-variadic function
    fn t_f6(..., x: isize);
    //~^ ERROR defining functions with C-variadic arguments is only allowed for free functions with the "C" or "C-unwind" calling convention
    //~| ERROR `...` must be the last argument of a C-variadic function
}
