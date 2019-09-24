// aux-build:ident-mac.rs

#![feature(c_variadic)]

extern crate ident_mac;
use ident_mac::id;

struct W(u8);

extern "C" { fn ffi(#[id] arg1: i32, #[id] ...); }
//~^ ERROR expected an inert attribute, found an attribute macro
//~| ERROR expected an inert attribute, found an attribute macro

unsafe extern "C" fn cvar(arg1: i32, #[id] mut args: ...) {}
//~^ ERROR expected an inert attribute, found an attribute macro

type Alias = extern "C" fn(#[id] u8, #[id] ...);
    //~^ ERROR expected an inert attribute, found an attribute macro
    //~| ERROR expected an inert attribute, found an attribute macro

fn free(#[id] arg1: u8) {
    //~^ ERROR expected an inert attribute, found an attribute macro
    let lam = |#[id] W(x), #[id] y| ();
    //~^ ERROR expected an inert attribute, found an attribute macro
    //~| ERROR expected an inert attribute, found an attribute macro
}

impl W {
    fn inherent1(#[id] self, #[id] arg1: u8) {}
    //~^ ERROR expected an inert attribute, found an attribute macro
    //~| ERROR expected an inert attribute, found an attribute macro
    fn inherent2(#[id] &self, #[id] arg1: u8) {}
    //~^ ERROR expected an inert attribute, found an attribute macro
    //~| ERROR expected an inert attribute, found an attribute macro
    fn inherent3<'a>(#[id] &'a mut self, #[id] arg1: u8) {}
    //~^ ERROR expected an inert attribute, found an attribute macro
    //~| ERROR expected an inert attribute, found an attribute macro
    fn inherent4<'a>(#[id] self: Box<Self>, #[id] arg1: u8) {}
    //~^ ERROR expected an inert attribute, found an attribute macro
    //~| ERROR expected an inert attribute, found an attribute macro
}

trait A {
    fn trait1(#[id] self, #[id] arg1: u8);
    //~^ ERROR expected an inert attribute, found an attribute macro
    //~| ERROR expected an inert attribute, found an attribute macro
    fn trait2(#[id] &self, #[id] arg1: u8);
    //~^ ERROR expected an inert attribute, found an attribute macro
    //~| ERROR expected an inert attribute, found an attribute macro
    fn trait3<'a>(#[id] &'a mut self, #[id] arg1: u8);
    //~^ ERROR expected an inert attribute, found an attribute macro
    //~| ERROR expected an inert attribute, found an attribute macro
    fn trait4<'a>(#[id] self: Box<Self>, #[id] arg1: u8, #[id] Vec<u8>);
    //~^ ERROR expected an inert attribute, found an attribute macro
    //~| ERROR expected an inert attribute, found an attribute macro
    //~| ERROR expected an inert attribute, found an attribute macro
}

fn main() {}
