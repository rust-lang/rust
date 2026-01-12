//@ edition: 2024
#![allow(internal_features, unused_imports, unused_macros)]
#![feature(extern_types)]
#![feature(gen_blocks)]
#![feature(rustc_attrs)]
#![feature(stmt_expr_attributes)]
#![feature(trait_alias)]

#[rustc_scalable_vector(4)]
//~^ ERROR: `#[rustc_scalable_vector]` attribute cannot be used on extern crates
extern crate std as other_std;

#[rustc_scalable_vector(4)]
//~^ ERROR: `#[rustc_scalable_vector]` attribute cannot be used on use statements
use std::vec::Vec;

#[rustc_scalable_vector(4)]
//~^ ERROR: `#[rustc_scalable_vector]` attribute cannot be used on statics
static _X: u32 = 0;

#[rustc_scalable_vector(4)]
//~^ ERROR: `#[rustc_scalable_vector]` attribute cannot be used on constants
const _Y: u32 = 0;

#[rustc_scalable_vector(4)]
//~^ ERROR: `#[rustc_scalable_vector]` attribute cannot be used on modules
mod bar {
}

#[rustc_scalable_vector(4)]
//~^ ERROR: `#[rustc_scalable_vector]` attribute cannot be used on foreign modules
unsafe extern "C" {
    #[rustc_scalable_vector(4)]
//~^ ERROR: `#[rustc_scalable_vector]` attribute cannot be used on foreign statics
    static X: &'static u32;
    #[rustc_scalable_vector(4)]
//~^ ERROR: `#[rustc_scalable_vector]` attribute cannot be used on foreign types
    type Y;
    #[rustc_scalable_vector(4)]
//~^ ERROR: `#[rustc_scalable_vector]` attribute cannot be used on foreign functions
    fn foo();
}

#[rustc_scalable_vector(4)]
//~^ ERROR: `#[rustc_scalable_vector]` attribute cannot be used on type aliases
type Foo = u32;

#[rustc_scalable_vector(4)]
//~^ ERROR: `#[rustc_scalable_vector]` attribute cannot be used on enums
enum Bar<#[rustc_scalable_vector(4)] T> {
//~^ ERROR: `#[rustc_scalable_vector]` attribute cannot be used on type parameters
    #[rustc_scalable_vector(4)]
//~^ ERROR: `#[rustc_scalable_vector]` attribute cannot be used on enum variants
    Baz(std::marker::PhantomData<T>),
}

struct Qux {
    #[rustc_scalable_vector(4)]
//~^ ERROR: `#[rustc_scalable_vector]` attribute cannot be used on struct fields
    field: u32,
}

#[rustc_scalable_vector(4)]
//~^ ERROR: `#[rustc_scalable_vector]` attribute cannot be used on unions
union FooBar {
    x: u32,
    y: u32,
}

#[rustc_scalable_vector(4)]
//~^ ERROR: `#[rustc_scalable_vector]` attribute cannot be used on traits
trait FooBaz {
    #[rustc_scalable_vector(4)]
//~^ ERROR: `#[rustc_scalable_vector]` attribute cannot be used on associated types
    type Foo;
    #[rustc_scalable_vector(4)]
//~^ ERROR: `#[rustc_scalable_vector]` attribute cannot be used on associated consts
    const Bar: i32;
    #[rustc_scalable_vector(4)]
//~^ ERROR: `#[rustc_scalable_vector]` attribute cannot be used on provided trait methods
    fn foo() {}
}

#[rustc_scalable_vector(4)]
//~^ ERROR: `#[rustc_scalable_vector]` attribute cannot be used on trait aliases
trait FooQux = FooBaz;

#[rustc_scalable_vector(4)]
//~^ ERROR: `#[rustc_scalable_vector]` attribute cannot be used on inherent impl blocks
impl<T> Bar<T> {
    #[rustc_scalable_vector(4)]
//~^ ERROR: `#[rustc_scalable_vector]` attribute cannot be used on inherent methods
    fn foo() {}
}

#[rustc_scalable_vector(4)]
//~^ ERROR: `#[rustc_scalable_vector]` attribute cannot be used on trait impl blocks
impl<T> FooBaz for Bar<T> {
    type Foo = u32;
    const Bar: i32 = 3;
}

#[rustc_scalable_vector(4)]
//~^ ERROR: `#[rustc_scalable_vector]` attribute cannot be used on macro defs
macro_rules! barqux { ($foo:tt) => { $foo }; }

#[rustc_scalable_vector(4)]
//~^ ERROR: `#[rustc_scalable_vector]` attribute cannot be used on functions
fn barqux(#[rustc_scalable_vector(4)] _x: u32) {}
//~^ ERROR: `#[rustc_scalable_vector]` attribute cannot be used on function params
//~^^ ERROR: allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters

#[rustc_scalable_vector(4)]
//~^ ERROR: `#[rustc_scalable_vector]` attribute cannot be used on functions
async fn async_foo() {}

#[rustc_scalable_vector(4)]
//~^ ERROR: `#[rustc_scalable_vector]` attribute cannot be used on functions
gen fn gen_foo() {}

#[rustc_scalable_vector(4)]
//~^ ERROR: `#[rustc_scalable_vector]` attribute cannot be used on functions
async gen fn async_gen_foo() {}

fn main() {
    let _x = #[rustc_scalable_vector(4)] || { };
//~^ ERROR: `#[rustc_scalable_vector]` attribute cannot be used on closures
    let _y = #[rustc_scalable_vector(4)] 3 + 4;
//~^ ERROR: `#[rustc_scalable_vector]` attribute cannot be used on expressions
    #[rustc_scalable_vector(4)]
//~^ ERROR: `#[rustc_scalable_vector]` attribute cannot be used on statements
    let _z = 3;

    match _z {
        #[rustc_scalable_vector(4)]
//~^ ERROR: `#[rustc_scalable_vector]` attribute cannot be used on match arms
        1 => (),
        _ => (),
    }
}

#[rustc_scalable_vector("4")]
//~^ ERROR: malformed `rustc_scalable_vector` attribute input
struct ArgNotLit(f32);

#[rustc_scalable_vector(4, 2)]
//~^ ERROR: malformed `rustc_scalable_vector` attribute input
struct ArgMultipleLits(f32);

#[rustc_scalable_vector(count = "4")]
//~^ ERROR: malformed `rustc_scalable_vector` attribute input
struct ArgKind(f32);

#[rustc_scalable_vector(65536)]
//~^ ERROR: element count in `rustc_scalable_vector` is too large: `65536`
struct CountTooLarge(f32);

#[rustc_scalable_vector(4)]
struct Okay(f32);

#[rustc_scalable_vector]
struct OkayNoArg(f32);
//~^ ERROR: scalable vector structs can only have scalable vector fields
