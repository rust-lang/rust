//@ compile-flags: --crate-type=lib
#![feature(repr_simd, repr_scalable)]

#[repr(simd, scalable(4))]
pub struct ScalableFloat {
    _ty: [f32]
}

trait WithAssocTy {
    type Ty;
}

impl WithAssocTy for ScalableFloat {
    type Ty = Self;
}

pub enum AsEnumField {
    Scalable(ScalableFloat), //~ ERROR: scalable vectors cannot be fields of a variant
}

pub struct AsStructField {
    v: ScalableFloat, //~ ERROR: scalable vectors cannot be fields of a struct
}

pub union AsUnionField {
    v: ScalableFloat,
//~^ ERROR: scalable vectors cannot be fields of a union
//~^^ ERROR: field must implement `Copy` or be wrapped in `ManuallyDrop<...>` to be used in a union
}

pub enum IndirectAsEnumField {
    Scalable(<ScalableFloat as WithAssocTy>::Ty), //~ ERROR: scalable vectors cannot be fields of a variant
}

pub struct IndirectAsStructField {
    v: <ScalableFloat as WithAssocTy>::Ty, //~ ERROR: scalable vectors cannot be fields of a struct
}

pub union IndirectAsUnionField {
    v: <ScalableFloat as WithAssocTy>::Ty,
//~^ ERROR: scalable vectors cannot be fields of a union
//~^^ ERROR: field must implement `Copy` or be wrapped in `ManuallyDrop<...>` to be used in a union
}

fn foo() {
    let x: [ScalableFloat; 2]; //~ ERROR: scalable vectors cannot be array elements
    let y: (ScalableFloat, u32); //~ ERROR: scalable vectors cannot be tuple fields
    let z: (u32, ScalableFloat); //~ ERROR: scalable vectors cannot be tuple fields

    // FIXME(repr-scalable): these should error too
    let indirect_x: [<ScalableFloat as WithAssocTy>::Ty; 2];
    let indirect_y: (<ScalableFloat as WithAssocTy>::Ty, u32);
    let indirect_z: (u32, <ScalableFloat as WithAssocTy>::Ty);
}
