fn f1(_: fn(a: u8)) {}
fn f2(_: impl Fn(u8, vvvv: u8)) {} //~ ERROR parameters in parenthesized generic argument lists with names, attributes, or c-variadics are experimental
fn f3(_: impl Fn(aaaa: u8, u8)) {} //~ ERROR parameters in parenthesized generic argument lists with names, attributes, or c-variadics are experimental
fn f4(_: impl Fn(aaaa: u8, vvvv: u8)) {}
//~^ ERROR parameters in parenthesized generic argument lists with names, attributes, or c-variadics are experimental
//~| ERROR parameters in parenthesized generic argument lists with names, attributes, or c-variadics are experimental
fn f5(_: impl Fn(u8, ...)) {}
//~^ ERROR parameters in parenthesized generic argument lists with names, attributes, or c-variadics are experimental
//~| ERROR C-variadic parameters aren't allowed in parenthesized argument lists
fn f6(f: impl Fn(#[cfg(false)] u8)) { f() } //~ ERROR parameters in parenthesized generic argument lists with names, attributes, or c-variadics are experimental
fn f7(_: impl Fn(#[inline] u8)) {}
//~^ ERROR parameters in parenthesized generic argument lists with names, attributes, or c-variadics are experimental
//~| ERROR the `inline` attribute cannot be used on function params
//~| ERROR: allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters
fn f8(_: impl Fn(u8, #[allow(unused_attributes)] u8)) {}
//~^ ERROR parameters in parenthesized generic argument lists with names, attributes, or c-variadics are experimental


fn main(){}
