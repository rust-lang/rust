// aux-build:ident-mac.rs

#![feature(param_attrs)]
#![feature(c_variadic)]

extern crate ident_mac;
use ident_mac::id;

struct W(u8);

extern "C" { fn ffi(#[id] arg1: i32, #[id] ...); }
//~^ ERROR the attribute `id` is currently unknown to the compiler
//~| ERROR the attribute `id` is currently unknown to the compiler

unsafe extern "C" fn cvar(arg1: i32, #[id] mut args: ...) {}
//~^ ERROR the attribute `id` is currently unknown to the compiler

type Alias = extern "C" fn(#[id] u8, #[id] ...);
    //~^ ERROR the attribute `id` is currently unknown to the compiler
    //~| ERROR the attribute `id` is currently unknown to the compiler

fn free(#[id] arg1: u8) {
    //~^ ERROR the attribute `id` is currently unknown to the compiler
    let lam = |#[id] W(x), #[id] y| ();
    //~^ ERROR the attribute `id` is currently unknown to the compiler
    //~| ERROR the attribute `id` is currently unknown to the compiler
}

impl W {
    fn inherent1(#[id] self, #[id] arg1: u8) {}
    //~^ ERROR the attribute `id` is currently unknown to the compiler
    //~| ERROR the attribute `id` is currently unknown to the compiler
    fn inherent2(#[id] &self, #[id] arg1: u8) {}
    //~^ ERROR the attribute `id` is currently unknown to the compiler
    //~| ERROR the attribute `id` is currently unknown to the compiler
    fn inherent3<'a>(#[id] &'a mut self, #[id] arg1: u8) {}
    //~^ ERROR the attribute `id` is currently unknown to the compiler
    //~| ERROR the attribute `id` is currently unknown to the compiler
    fn inherent4<'a>(#[id] self: Box<Self>, #[id] arg1: u8) {}
    //~^ ERROR the attribute `id` is currently unknown to the compiler
    //~| ERROR the attribute `id` is currently unknown to the compiler
}

trait A {
    fn trait1(#[id] self, #[id] arg1: u8);
    //~^ ERROR the attribute `id` is currently unknown to the compiler
    //~| ERROR the attribute `id` is currently unknown to the compiler
    fn trait2(#[id] &self, #[id] arg1: u8);
    //~^ ERROR the attribute `id` is currently unknown to the compiler
    //~| ERROR the attribute `id` is currently unknown to the compiler
    fn trait3<'a>(#[id] &'a mut self, #[id] arg1: u8);
    //~^ ERROR the attribute `id` is currently unknown to the compiler
    //~| ERROR the attribute `id` is currently unknown to the compiler
    fn trait4<'a>(#[id] self: Box<Self>, #[id] arg1: u8, #[id] Vec<u8>);
    //~^ ERROR the attribute `id` is currently unknown to the compiler
    //~| ERROR the attribute `id` is currently unknown to the compiler
    //~| ERROR the attribute `id` is currently unknown to the compiler
}

fn main() {}
