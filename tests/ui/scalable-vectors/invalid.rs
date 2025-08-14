//@ edition: 2024
#![allow(internal_features, unused_imports, unused_macros)]
#![feature(extern_types)]
#![feature(gen_blocks)]
#![feature(rustc_attrs)]
#![feature(stmt_expr_attributes)]
#![feature(trait_alias)]

#[rustc_scalable_vector(4)]
//~^ ERROR: attribute should be applied to a struct
extern crate std as other_std;

#[rustc_scalable_vector(4)]
//~^ ERROR: attribute should be applied to a struct
use std::vec::Vec;

#[rustc_scalable_vector(4)]
//~^ ERROR: attribute should be applied to a struct
static _X: u32 = 0;

#[rustc_scalable_vector(4)]
//~^ ERROR: attribute should be applied to a struct
const _Y: u32 = 0;

#[rustc_scalable_vector(4)]
//~^ ERROR: attribute should be applied to a struct
mod bar {
}

#[rustc_scalable_vector(4)]
//~^ ERROR: attribute should be applied to a struct
unsafe extern "C" {
    #[rustc_scalable_vector(4)]
//~^ ERROR: attribute should be applied to a struct
    static X: &'static u32;
    #[rustc_scalable_vector(4)]
//~^ ERROR: attribute should be applied to a struct
    type Y;
    #[rustc_scalable_vector(4)]
//~^ ERROR: attribute should be applied to a struct
    fn foo();
}

#[rustc_scalable_vector(4)]
//~^ ERROR: attribute should be applied to a struct
type Foo = u32;

#[rustc_scalable_vector(4)]
//~^ ERROR: attribute should be applied to a struct
enum Bar<#[rustc_scalable_vector(4)] T> {
//~^ ERROR: attribute should be applied to a struct
    #[rustc_scalable_vector(4)]
//~^ ERROR: attribute should be applied to a struct
    Baz(std::marker::PhantomData<T>),
}

struct Qux {
    #[rustc_scalable_vector(4)]
//~^ ERROR: attribute should be applied to a struct
    field: u32,
}

#[rustc_scalable_vector(4)]
//~^ ERROR: attribute should be applied to a struct
union FooBar {
    x: u32,
    y: u32,
}

#[rustc_scalable_vector(4)]
//~^ ERROR: attribute should be applied to a struct
trait FooBaz {
    #[rustc_scalable_vector(4)]
//~^ ERROR: attribute should be applied to a struct
    type Foo;
    #[rustc_scalable_vector(4)]
//~^ ERROR: attribute should be applied to a struct
    const Bar: i32;
    #[rustc_scalable_vector(4)]
//~^ ERROR: attribute should be applied to a struct
    fn foo() {}
}

#[rustc_scalable_vector(4)]
//~^ ERROR: attribute should be applied to a struct
trait FooQux = FooBaz;

#[rustc_scalable_vector(4)]
//~^ ERROR: attribute should be applied to a struct
impl<T> Bar<T> {
    #[rustc_scalable_vector(4)]
//~^ ERROR: attribute should be applied to a struct
    fn foo() {}
}

#[rustc_scalable_vector(4)]
//~^ ERROR: attribute should be applied to a struct
impl<T> FooBaz for Bar<T> {
    type Foo = u32;
    const Bar: i32 = 3;
}

#[rustc_scalable_vector(4)]
//~^ ERROR: attribute should be applied to a struct
macro_rules! barqux { ($foo:tt) => { $foo }; }

#[rustc_scalable_vector(4)]
//~^ ERROR: attribute should be applied to a struct
fn barqux(#[rustc_scalable_vector(4)] _x: u32) {}
//~^ ERROR: attribute should be applied to a struct
//~^^ ERROR: allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters

#[rustc_scalable_vector(4)]
//~^ ERROR: attribute should be applied to a struct
async fn async_foo() {}

#[rustc_scalable_vector(4)]
//~^ ERROR: attribute should be applied to a struct
gen fn gen_foo() {}

#[rustc_scalable_vector(4)]
//~^ ERROR: attribute should be applied to a struct
async gen fn async_gen_foo() {}

fn main() {
    let _x = #[rustc_scalable_vector(4)] || { };
//~^ ERROR: attribute should be applied to a struct
    let _y = #[rustc_scalable_vector(4)] 3 + 4;
//~^ ERROR: attribute should be applied to a struct
    #[rustc_scalable_vector(4)]
//~^ ERROR: attribute should be applied to a struct
    let _z = 3;

    match _z {
        #[rustc_scalable_vector(4)]
//~^ ERROR: attribute should be applied to a struct
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

#[rustc_scalable_vector(4)]
struct Okay(f32);

#[rustc_scalable_vector]
struct OkayNoArg(f32);
