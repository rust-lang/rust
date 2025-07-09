//@ edition: 2024
#![allow(internal_features, unused_imports, unused_macros)]
#![feature(extern_types)]
#![feature(gen_blocks)]
#![feature(repr_scalable)]
#![feature(repr_simd)]
#![feature(stmt_expr_attributes)]
#![feature(trait_alias)]

#[repr(simd, scalable(4))]
//~^ ERROR: attribute should be applied to a struct
//~^^ ERROR: attribute should be applied to a struct
extern crate std as other_std;

#[repr(simd, scalable(4))]
//~^ ERROR: attribute should be applied to a struct
//~^^ ERROR: attribute should be applied to a struct
use std::vec::Vec;

#[repr(simd, scalable(4))]
//~^ ERROR: attribute should be applied to a struct
//~^^ ERROR: attribute should be applied to a struct
static _X: u32 = 0;

#[repr(simd, scalable(4))]
//~^ ERROR: attribute should be applied to a struct
//~^^ ERROR: attribute should be applied to a struct
const _Y: u32 = 0;

#[repr(simd, scalable(4))]
//~^ ERROR: attribute should be applied to a struct
//~^^ ERROR: attribute should be applied to a struct
mod bar {
}

#[repr(simd, scalable(4))]
//~^ ERROR: attribute should be applied to a struct
//~^^ ERROR: attribute should be applied to a struct
unsafe extern "C" {
    #[repr(simd, scalable(4))]
//~^ ERROR: attribute should be applied to a struct
//~^^ ERROR: attribute should be applied to a struct
    static X: &'static u32;
    #[repr(simd, scalable(4))]
//~^ ERROR: attribute should be applied to a struct
//~^^ ERROR: attribute should be applied to a struct
    type Y;
    #[repr(simd, scalable(4))]
//~^ ERROR: attribute should be applied to a struct
//~^^ ERROR: attribute should be applied to a struct
    fn foo();
}

#[repr(simd, scalable(4))]
//~^ ERROR: attribute should be applied to a struct
//~^^ ERROR: attribute should be applied to a struct
type Foo = u32;

#[repr(simd, scalable(4))]
//~^ ERROR: attribute should be applied to a struct
//~^^ ERROR: attribute should be applied to a struct
enum Bar<#[repr(simd, scalable(4))] T> {
//~^ ERROR: attribute should be applied to a struct
//~^^ ERROR: attribute should be applied to a struct
    #[repr(simd, scalable(4))]
//~^ ERROR: attribute should be applied to a struct
//~^^ ERROR: attribute should be applied to a struct
    Baz(std::marker::PhantomData<T>),
}

struct Qux {
    #[repr(simd, scalable(4))]
//~^ ERROR: attribute should be applied to a struct
//~^^ ERROR: attribute should be applied to a struct
    field: u32,
}

#[repr(simd, scalable(4))]
//~^ ERROR: attribute should be applied to a struct
//~^^ ERROR: attribute should be applied to a struct
union FooBar {
    x: u32,
    y: u32,
}

#[repr(simd, scalable(4))]
//~^ ERROR: attribute should be applied to a struct
//~^^ ERROR: attribute should be applied to a struct
trait FooBaz {
    #[repr(simd, scalable(4))]
//~^ ERROR: attribute should be applied to a struct
//~^^ ERROR: attribute should be applied to a struct
    type Foo;
    #[repr(simd, scalable(4))]
//~^ ERROR: attribute should be applied to a struct
//~^^ ERROR: attribute should be applied to a struct
    const Bar: i32;
    #[repr(simd, scalable(4))]
//~^ ERROR: attribute should be applied to a struct
//~^^ ERROR: attribute should be applied to a struct
    fn foo() {}
}

#[repr(simd, scalable(4))]
//~^ ERROR: attribute should be applied to a struct
//~^^ ERROR: attribute should be applied to a struct
trait FooQux = FooBaz;

#[repr(simd, scalable(4))]
//~^ ERROR: attribute should be applied to a struct
//~^^ ERROR: attribute should be applied to a struct
impl<T> Bar<T> {
    #[repr(simd, scalable(4))]
//~^ ERROR: attribute should be applied to a struct
//~^^ ERROR: attribute should be applied to a struct
    fn foo() {}
}

#[repr(simd, scalable(4))]
//~^ ERROR: attribute should be applied to a struct
//~^^ ERROR: attribute should be applied to a struct
impl<T> FooBaz for Bar<T> {
    type Foo = u32;
    const Bar: i32 = 3;
}

#[repr(simd, scalable(4))]
//~^ ERROR: attribute should be applied to a struct
//~^^ ERROR: attribute should be applied to a struct
macro_rules! barqux { ($foo:tt) => { $foo }; }

#[repr(simd, scalable(4))]
//~^ ERROR: attribute should be applied to a struct
//~^^ ERROR: attribute should be applied to a struct
fn barqux(#[repr(simd, scalable(4))] _x: u32) {}
//~^ ERROR: attribute should be applied to a struct
//~^^ ERROR: attribute should be applied to a struct
//~^^^ ERROR: allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters

#[repr(simd, scalable(4))]
//~^ ERROR: attribute should be applied to a struct
//~^^ ERROR: attribute should be applied to a struct
async fn async_foo() {}

#[repr(simd, scalable(4))]
//~^ ERROR: attribute should be applied to a struct
//~^^ ERROR: attribute should be applied to a struct
gen fn gen_foo() {}

#[repr(simd, scalable(4))]
//~^ ERROR: attribute should be applied to a struct
//~^^ ERROR: attribute should be applied to a struct
async gen fn async_gen_foo() {}

fn main() {
    let _x = #[repr(simd, scalable(4))] || { };
//~^ ERROR: attribute should be applied to a struct
//~^^ ERROR: attribute should be applied to a struct
    let _y = #[repr(simd, scalable(4))] 3 + 4;
//~^ ERROR: attribute should be applied to a struct
//~^^ ERROR: attribute should be applied to a struct
    #[repr(simd, scalable(4))]
//~^ ERROR: attribute should be applied to a struct
//~^^ ERROR: attribute should be applied to a struct
    let _z = 3;

    match _z {
        #[repr(simd, scalable(4))]
//~^ ERROR: attribute should be applied to a struct
//~^^ ERROR: attribute should be applied to a struct
        1 => (),
        _ => (),
    }
}

#[repr(transparent, simd, scalable(4))] //~ ERROR: transparent struct cannot have other repr hints
struct CombinedWithReprTransparent {
    _ty: [f64],
}

#[repr(Rust, simd, scalable(4))] //~ ERROR: conflicting representation hints
struct CombinedWithReprRust {
    _ty: [f64],
}

#[repr(C, simd, scalable(4))]
//~^ ERROR: conflicting representation hints
//~^^ WARN: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
struct CombinedWithReprC {
    _ty: [f64],
}

#[repr(scalable(4))] //~ ERROR: `scalable` representation hint without `simd` representation hint
struct WithoutReprSimd {
    _ty: [f64],
}

#[repr(simd, scalable)] //~ ERROR: invalid `scalable(num)` attribute: `scalable` needs an argument
struct MissingArg {
//~^ ERROR: SIMD vector's only field must be an array
    _ty: [f64],
}

#[repr(simd, scalable("4"))] //~ ERROR: invalid `scalable(num)` attribute: `scalable` needs an argument
struct ArgNotLit {
//~^ ERROR: SIMD vector's only field must be an array
    _ty: [f64],
}

#[repr(simd, scalable(4, 2))] //~ ERROR: invalid `scalable(num)` attribute: `scalable` needs an argument
struct ArgMultipleLits {
//~^ ERROR: SIMD vector's only field must be an array
    _ty: [f64],
}

#[repr(simd, scalable = "4")] //~ ERROR: unrecognized representation hint
struct ArgKind {
//~^ ERROR: SIMD vector's only field must be an array
    _ty: [f64],
}

#[repr(simd, scalable(4))]
struct Okay {
    _ty: [f64],
}
