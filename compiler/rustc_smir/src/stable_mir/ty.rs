use super::{mir::Mutability, with, DefId};
use crate::rustc_internal::Opaque;

#[derive(Copy, Clone, Debug)]
pub struct Ty(pub usize);

impl Ty {
    pub fn kind(&self) -> TyKind {
        with(|context| context.ty_kind(*self))
    }
}

type Const = Opaque;
pub(crate) type Region = Opaque;

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
    Adt(AdtDef, GenericArgs),
    Foreign(ForeignDef),
    Str,
    Array(Ty, Const),
    Slice(Ty),
    RawPtr(Ty, Mutability),
    Ref(Region, Ty, Mutability),
    FnDef(FnDef, GenericArgs),
    Closure(ClosureDef, GenericArgs),
    Generator(GeneratorDef, GenericArgs, Movability),
    Never,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Movability {
    Static,
    Movable,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ForeignDef(pub(crate) DefId);

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FnDef(pub(crate) DefId);

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ClosureDef(pub(crate) DefId);

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct GeneratorDef(pub(crate) DefId);

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct AdtDef(pub(crate) DefId);

#[derive(Clone, Debug)]
pub struct GenericArgs(pub Vec<GenericArgKind>);

#[derive(Clone, Debug)]
pub enum GenericArgKind {
    Lifetime(Region),
    Type(Ty),
    Const(Const),
}
