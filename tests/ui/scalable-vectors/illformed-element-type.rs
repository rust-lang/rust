//@ compile-flags: --crate-type=lib
#![allow(internal_features)]
#![feature(extern_types)]
#![feature(never_type)]
#![feature(rustc_attrs)]

struct Foo;
enum Bar {}
union Baz { x: u16 }
extern "C" {
    type Qux;
}

#[rustc_scalable_vector(4)]
struct TyChar(char);
//~^ ERROR: element type of a scalable vector must be a primitive scalar

#[rustc_scalable_vector(2)]
struct TyConstPtr(*const u8);
//~^ ERROR: element type of a scalable vector must be a primitive scalar

#[rustc_scalable_vector(2)]
struct TyMutPtr(*mut u8);
//~^ ERROR: element type of a scalable vector must be a primitive scalar

#[rustc_scalable_vector(4)]
struct TyStruct(Foo);
//~^ ERROR: element type of a scalable vector must be a primitive scalar

#[rustc_scalable_vector(4)]
struct TyEnum(Bar);
//~^ ERROR: element type of a scalable vector must be a primitive scalar

#[rustc_scalable_vector(4)]
struct TyUnion(Baz);
//~^ ERROR: element type of a scalable vector must be a primitive scalar

#[rustc_scalable_vector(4)]
struct TyForeign(Qux);
//~^ ERROR: element type of a scalable vector must be a primitive scalar

#[rustc_scalable_vector(4)]
struct TyArray([u32; 4]);
//~^ ERROR: element type of a scalable vector must be a primitive scalar

#[rustc_scalable_vector(4)]
struct TySlice([u32]);
//~^ ERROR: element type of a scalable vector must be a primitive scalar

#[rustc_scalable_vector(4)]
struct TyRef<'a>(&'a u32);
//~^ ERROR: element type of a scalable vector must be a primitive scalar

#[rustc_scalable_vector(4)]
struct TyFnPtr(fn(u32) -> u32);
//~^ ERROR: element type of a scalable vector must be a primitive scalar

#[rustc_scalable_vector(4)]
struct TyDyn(dyn std::io::Write);
//~^ ERROR: element type of a scalable vector must be a primitive scalar

#[rustc_scalable_vector(4)]
struct TyNever(!);
//~^ ERROR: element type of a scalable vector must be a primitive scalar

#[rustc_scalable_vector(4)]
struct TyTuple((u32, u32));
//~^ ERROR: element type of a scalable vector must be a primitive scalar

type ValidAlias = u32;
type InvalidAlias = String;

#[rustc_scalable_vector(4)]
struct TyValidAlias(ValidAlias);

#[rustc_scalable_vector(4)]
struct TyInvalidAlias(InvalidAlias);
//~^ ERROR: element type of a scalable vector must be a primitive scalar

trait Tr {
    type Valid;
    type Invalid;
}

impl Tr for () {
    type Valid = u32;
    type Invalid = String;
}

struct TyValidProjection(<() as Tr>::Valid);

struct TyInvalidProjection(<() as Tr>::Invalid);
// FIXME: element type of a scalable vector must be a primitive scalar
