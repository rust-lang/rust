#![feature(named_fn_trait_parameters)]

fn f1(f: impl Fn(#[cfg(false)] x: u8)) { f() }
fn f2(_: impl Fn(#[inline] x: u8)) {}
//~^ ERROR the `inline` attribute cannot be used on function params
//~| ERROR: allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters
fn f3(_: impl Fn(#[cfg_attr(true, inline)] a1: usize)) {}
//~^ ERROR the `inline` attribute cannot be used on function params
//~| ERROR: allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters
fn f4(_: impl Fn(/** */ x: u8)) {}
//~^ ERROR documentation comments cannot be applied to function parameters

fn f5(_: impl Fn(#[forbid(non_snake_case)] NON_SNAKE: u8)) {}
fn f6(_: impl Fn(#[forbid(keyword_idents_2024)] gen: u8)) {}
//~^ ERROR `gen` is a keyword in the 2024 edition [keyword_idents_2024]
//~| WARN this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2024!
fn f7(_: impl Fn(#[allow(keyword_idents_2024)] gen: u8)) {}
fn f8(_: fn(#[rustc_splat] a: usize)) {}
//~^ ERROR the `rustc_splat` attribute is an experimental feature



fn main(){}
