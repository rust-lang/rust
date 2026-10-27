fn f1(_: fn(a: u8)) {}
fn f2(_: impl Fn(u8, vvvv: u8)) {} //~ ERROR named parameters in parenthesized generic argument lists are experimental
fn f3(_: impl Fn(aaaa: u8, u8)) {} //~ ERROR named parameters in parenthesized generic argument lists are experimental
fn f4(_: impl Fn(aaaa: u8, vvvv: u8)) {}
//~^ ERROR named parameters in parenthesized generic argument lists are experimental
//~| ERROR named parameters in parenthesized generic argument lists are experimental
fn f5(_: impl Fn(u8, ...)) {}
//~^ ERROR `Trait(...)` syntax does not support c_variadic parameters
fn f6(f: impl Fn(#[cfg(false)] u8)) { f() } //~ ERROR named parameters in parenthesized generic argument lists are experimental
fn f7(_: impl Fn(#[inline] u8)) {}
//~^ ERROR named parameters in parenthesized generic argument lists are experimental
//~| ERROR the `inline` attribute cannot be used on function params
//~| ERROR: allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters
fn f8(_: impl Fn(u8, #[allow(unused_attributes)] u8)) {}
//~^ ERROR named parameters in parenthesized generic argument lists are experimental


fn main(){}
