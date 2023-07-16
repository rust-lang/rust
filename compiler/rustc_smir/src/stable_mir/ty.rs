use super::{with, DefId};
use crate::rustc_internal::Opaque;

#[derive(Copy, Clone, Debug)]
pub struct Ty(pub usize);

impl Ty {
    pub fn kind(&self) -> TyKind {
        with(|context| context.ty_kind(*self))
    }
}

type Const = Opaque;
type Region = Opaque;

#[derive(Clone, Debug)]
pub enum TyKind {
    RigidTy(RigidTy),
}

#[derive(Clone, Debug)]
pub enum RigidTy {
    Bool,
    Char,
    Int(IntTy),
    Uint(UintTy),
    Float(FloatTy),
    Adt(AdtDef, AdtSubsts),
    Str,
    Array(Ty, Const),
    Slice(Ty),
    Tuple(Vec<Ty>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntTy {
    Isize,
    I8,
    I16,
    I32,
    I64,
    I128,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UintTy {
    Usize,
    U8,
    U16,
    U32,
    U64,
    U128,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloatTy {
    F32,
    F64,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct AdtDef(pub(crate) DefId);

#[derive(Clone, Debug)]
pub struct AdtSubsts(pub Vec<GenericArgKind>);

#[derive(Clone, Debug)]
pub enum GenericArgKind {
    // FIXME add proper region
    Lifetime(Region),
    Type(Ty),
    // FIXME add proper const
    Const(Const),
}
