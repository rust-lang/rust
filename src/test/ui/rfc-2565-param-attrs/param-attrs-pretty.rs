// aux-build:param-attrs.rs

// check-pass

#![feature(c_variadic)]

extern crate param_attrs;

use param_attrs::*;

struct W(u8);

#[attr_extern]
extern "C" { fn ffi(#[a1] arg1: i32, #[a2] ...); }

#[attr_extern_cvar]
unsafe extern "C" fn cvar(arg1: i32, #[a1] mut args: ...) {}

#[attr_alias]
type Alias = fn(#[a1] u8, #[a2] ...);

#[attr_free]
fn free(#[a1] arg1: u8) {
    let lam = |#[a2] W(x), #[a3] y| ();
}

impl W {
    #[attr_inherent_1]
    fn inherent1(#[a1] self, #[a2] arg1: u8) {}

    #[attr_inherent_2]
    fn inherent2(#[a1] &self, #[a2] arg1: u8) {}

    #[attr_inherent_3]
    fn inherent3<'a>(#[a1] &'a mut self, #[a2] arg1: u8) {}

    #[attr_inherent_4]
    fn inherent4<'a>(#[a1] self: Box<Self>, #[a2] arg1: u8) {}

    #[attr_inherent_issue_64682]
    fn inherent5(#[a1] #[a2] arg1: u8, #[a3] arg2: u8) {}
}

trait A {
    #[attr_trait_1]
    fn trait1(#[a1] self, #[a2] arg1: u8);

    #[attr_trait_2]
    fn trait2(#[a1] &self, #[a2] arg1: u8);

    #[attr_trait_3]
    fn trait3<'a>(#[a1] &'a mut self, #[a2] arg1: u8);

    #[attr_trait_4]
    fn trait4<'a>(#[a1] self: Box<Self>, #[a2] arg1: u8, #[a3] Vec<u8>);

    #[attr_trait_issue_64682]
    fn trait5(#[a1] #[a2] arg1: u8, #[a3] arg2: u8);
}

fn main() {}
