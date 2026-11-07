//@ run-rustfix

#![allow(unused)]

pub const P1: const* u8 = 0 as _;
//~^ ERROR: raw pointer types must be written as `*const T`
//~| HELP: put the `*` before `const`

pub const P2: mut* u8 = 1 as _;
//~^ ERROR: raw pointer types must be written as `*mut T`
//~| HELP: put the `*` before `mut`

pub const P3: const* i32 = std::ptr::null();
//~^ ERROR: raw pointer types must be written as `*const T`
//~| HELP: put the `*` before `const`

pub const P4: const* i32 = std::ptr::null();
//~^ ERROR: raw pointer types must be written as `*const T`
//~| HELP: put the `*` before `const`

pub const P5: mut* i32 = std::ptr::null_mut();
//~^ ERROR: raw pointer types must be written as `*mut T`
//~| HELP: put the `*` before `mut`

pub const P6: mut* i32 = std::ptr::null_mut();
//~^ ERROR: raw pointer types must be written as `*mut T`
//~| HELP: put the `*` before `mut`

pub const P7: const* Vec<u8> = std::ptr::null();
//~^ ERROR: raw pointer types must be written as `*const T`
//~| HELP: put the `*` before `const`

pub const P8: const* std::collections::HashMap<String, i32> = std::ptr::null();
//~^ ERROR: raw pointer types must be written as `*const T`
//~| HELP: put the `*` before `const`

fn func1(p: const* u8) {}
//~^ ERROR: raw pointer types must be written as `*const T`
//~| HELP: put the `*` before `const`

fn func2(p: mut* u8) {}
//~^ ERROR: raw pointer types must be written as `*mut T`
//~| HELP: put the `*` before `mut`

fn func3() -> const* u8 { std::ptr::null() }
//~^ ERROR: raw pointer types must be written as `*const T`
//~| HELP: put the `*` before `const`

fn func4() -> mut* u8 { std::ptr::null_mut() }
//~^ ERROR: raw pointer types must be written as `*mut T`
//~| HELP: put the `*` before `mut`

struct S1 {
    field: const* u8,
    //~^ ERROR: raw pointer types must be written as `*const T`
    //~| HELP: put the `*` before `const`
}

struct S2 {
    field: mut* u8,
    //~^ ERROR: raw pointer types must be written as `*mut T`
    //~| HELP: put the `*` before `mut`
}

type Tuple1 = (const* u8, i32);
//~^ ERROR: raw pointer types must be written as `*const T`
//~| HELP: put the `*` before `const`

type Tuple2 = (mut* u8, i32);
//~^ ERROR: raw pointer types must be written as `*mut T`
//~| HELP: put the `*` before `mut`

type Array1 = [const* u8; 10];
//~^ ERROR: raw pointer types must be written as `*const T`
//~| HELP: put the `*` before `const`

type Array2 = [mut* u8; 10];
//~^ ERROR: raw pointer types must be written as `*mut T`
//~| HELP: put the `*` before `mut`

type Alias1 = const* u8;
//~^ ERROR: raw pointer types must be written as `*const T`
//~| HELP: put the `*` before `const`

type Alias2 = mut* u8;
//~^ ERROR: raw pointer types must be written as `*mut T`
//~| HELP: put the `*` before `mut`

pub const P9: const *u8 = std::ptr::null();
//~^ ERROR: raw pointer types must be written as `*const T`
//~| HELP: put the `*` before `const`

pub const P10: const  *  u8 = std::ptr::null();
//~^ ERROR: raw pointer types must be written as `*const T`
//~| HELP: put the `*` before `const`

impl S1 {
    fn method(self, size: const* u32) {}
    //~^ ERROR: raw pointer types must be written as `*const T`
    //~| HELP: put the `*` before `const`
}

trait Trait1 {
    fn method(p: const* u8);
    //~^ ERROR: raw pointer types must be written as `*const T`
    //~| HELP: put the `*` before `const`
}

fn generic_func<T>() -> const* T { std::ptr::null() }
//~^ ERROR: raw pointer types must be written as `*const T`
//~| HELP: put the `*` before `const`

fn main() {}
