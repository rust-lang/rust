//@ edition: 2024
#![allow(internal_features, unused_imports, unused_macros)]
#![feature(extern_types)]
#![feature(gen_blocks)]
#![feature(rustc_attrs)]
#![feature(stmt_expr_attributes)]
#![feature(trait_alias)]

// Test that invalid force inlining attributes error as expected.

#[rustc_force_inline("foo")]
pub fn forced1() {
}

#[rustc_force_inline(bar, baz)]
//~^ ERROR malformed `rustc_force_inline` attribute input
pub fn forced2() {
}

#[rustc_force_inline(2)]
//~^ ERROR malformed `rustc_force_inline` attribute input
pub fn forced3() {
}

#[rustc_force_inline = 2]
//~^ ERROR malformed `rustc_force_inline` attribute input
pub fn forced4() {
}

#[rustc_force_inline]
//~^ ERROR attribute cannot be used on
extern crate std as other_std;

#[rustc_force_inline]
//~^ ERROR attribute cannot be used on
use std::collections::HashMap;

#[rustc_force_inline]
//~^ ERROR attribute cannot be used on
static _FOO: &'static str = "FOO";

#[rustc_force_inline]
//~^ ERROR attribute cannot be used on
const _BAR: u32 = 3;

#[rustc_force_inline]
//~^ ERROR attribute cannot be used on
mod foo { }

#[rustc_force_inline]
//~^ ERROR attribute cannot be used on
unsafe extern "C" {
    #[rustc_force_inline]
//~^ ERROR attribute cannot be used on
    static X: &'static u32;

    #[rustc_force_inline]
//~^ ERROR attribute cannot be used on
    type Y;

    #[rustc_force_inline]
//~^ ERROR attribute cannot be used on
    fn foo();
}

#[rustc_force_inline]
//~^ ERROR attribute cannot be used on
type Foo = u32;

#[rustc_force_inline]
//~^ ERROR attribute cannot be used on
enum Bar<#[rustc_force_inline] T> {
//~^ ERROR attribute cannot be used on
    #[rustc_force_inline]
//~^ ERROR attribute cannot be used on
    Baz(std::marker::PhantomData<T>),
}

#[rustc_force_inline]
//~^ ERROR attribute cannot be used on
struct Qux {
    #[rustc_force_inline]
//~^ ERROR attribute cannot be used on
    field: u32,
}

#[rustc_force_inline]
//~^ ERROR attribute cannot be used on
union FooBar {
    x: u32,
    y: u32,
}

#[rustc_force_inline]
//~^ ERROR attribute cannot be used on
trait FooBaz {
    #[rustc_force_inline]
//~^ ERROR attribute cannot be used on
    type Foo;
    #[rustc_force_inline]
//~^ ERROR attribute cannot be used on
    const Bar: i32;

    #[rustc_force_inline]
//~^ ERROR attribute cannot be used on
    fn foo() {}
}

#[rustc_force_inline]
//~^ ERROR attribute cannot be used on
trait FooQux = FooBaz;

#[rustc_force_inline]
//~^ ERROR attribute cannot be used on
impl<T> Bar<T> {
    #[rustc_force_inline]
    fn foo() {}
}

#[rustc_force_inline]
//~^ ERROR attribute cannot be used on
impl<T> FooBaz for Bar<T> {
    type Foo = u32;
    const Bar: i32 = 3;
}

#[rustc_force_inline]
//~^ ERROR attribute cannot be used on
macro_rules! barqux { ($foo:tt) => { $foo }; }

fn barqux(#[rustc_force_inline] _x: u32) {}
//~^ ERROR allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters
//~^^ ERROR attribute cannot be used on

#[rustc_force_inline]
//~^ ERROR attribute cannot be applied to a `async`, `gen` or `async gen` function
async fn async_foo() {}

#[rustc_force_inline]
//~^ ERROR attribute cannot be applied to a `async`, `gen` or `async gen` function
gen fn gen_foo() {}

#[rustc_force_inline]
//~^ ERROR attribute cannot be applied to a `async`, `gen` or `async gen` function
async gen fn async_gen_foo() {}

fn main() {
    let _x = #[rustc_force_inline] || { };
//~^ ERROR attribute cannot be used on
    let _y = #[rustc_force_inline] 3 + 4;
//~^ ERROR attribute cannot be used on
    #[rustc_force_inline]
//~^ ERROR attribute cannot be used on
    let _z = 3;

    match _z {
        #[rustc_force_inline]
//~^ ERROR attribute cannot be used on
        1 => (),
        _ => (),
    }
}
