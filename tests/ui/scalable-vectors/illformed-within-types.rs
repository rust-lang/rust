//@ compile-flags: --crate-type=lib
#![allow(internal_features)]
#![feature(rustc_attrs)]

#[rustc_scalable_vector(2)]
struct ValidI64(i64);

struct Struct {
    x: ValidI64,
//~^ ERROR: scalable vectors cannot be fields of a struct
    in_tuple: (ValidI64,),
//~^ ERROR: scalable vectors cannot be tuple fields
}

struct TupleStruct(ValidI64);
//~^ ERROR: scalable vectors cannot be fields of a struct

enum Enum {
    StructVariant { _ty: ValidI64 },
//~^ ERROR: scalable vectors cannot be fields of a variant
    TupleVariant(ValidI64),
//~^ ERROR: scalable vectors cannot be fields of a variant
}
