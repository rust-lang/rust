//! HIR for references to types. Paths in these are not yet resolved. They can
//! be directly created from an ast::TypeRef, without further queries.

use std::fmt::Write;

use hir_expand::name::Name;
use intern::Symbol;
use la_arena::Idx;
use thin_vec::ThinVec;

use crate::{
    LifetimeParamId, TypeParamId,
    builtin_type::{BuiltinInt, BuiltinType, BuiltinUint},
    expr_store::{
        ExpressionStore,
        path::{GenericArg, Path},
    },
    hir::{ExprId, Literal},
};

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Mutability {
    Shared,
    Mut,
}

impl Mutability {
    pub fn from_mutable(mutable: bool) -> Mutability {
        if mutable { Mutability::Mut } else { Mutability::Shared }
    }

    pub fn as_keyword_for_ref(self) -> &'static str {
        match self {
            Mutability::Shared => "",
            Mutability::Mut => "mut ",
        }
    }

    pub fn as_keyword_for_ptr(self) -> &'static str {
        match self {
            Mutability::Shared => "const ",
            Mutability::Mut => "mut ",
        }
    }

    /// Returns `true` if the mutability is [`Mut`].
    ///
    /// [`Mut`]: Mutability::Mut
    #[must_use]
    pub fn is_mut(&self) -> bool {
        matches!(self, Self::Mut)
    }

    /// Returns `true` if the mutability is [`Shared`].
    ///
    /// [`Shared`]: Mutability::Shared
    #[must_use]
    pub fn is_shared(&self) -> bool {
        matches!(self, Self::Shared)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Rawness {
    RawPtr,
    Ref,
}

impl Rawness {
    pub fn from_raw(is_raw: bool) -> Rawness {
        if is_raw { Rawness::RawPtr } else { Rawness::Ref }
    }

    pub fn is_raw(&self) -> bool {
        matches!(self, Self::RawPtr)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
/// A `TypeRefId` that is guaranteed to always be `TypeRef::Path`. We use this for things like
/// impl's trait, that are always paths but need to be traced back to source code.
pub struct PathId(TypeRefId);

impl PathId {
    #[inline]
    pub fn from_type_ref_unchecked(type_ref: TypeRefId) -> Self {
        Self(type_ref)
    }

    #[inline]
    pub fn type_ref(self) -> TypeRefId {
        self.0
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TraitRef {
    pub path: PathId,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct FnType {
    pub params: Box<[(Option<Name>, TypeRefId)]>,
    pub is_varargs: bool,
    pub is_unsafe: bool,
    pub abi: Option<Symbol>,
}

impl FnType {
    #[inline]
    pub fn split_params_and_ret(&self) -> (&[(Option<Name>, TypeRefId)], TypeRefId) {
        let (ret, params) = self.params.split_last().expect("should have at least return type");
        (params, ret.1)
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ArrayType {
    pub ty: TypeRefId,
    pub len: ConstRef,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct RefType {
    pub ty: TypeRefId,
    pub lifetime: Option<LifetimeRefId>,
    pub mutability: Mutability,
}

/// Compare ty::Ty
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum TypeRef {
    Never,
    Placeholder,
    Tuple(ThinVec<TypeRefId>),
    Path(Path),
    RawPtr(TypeRefId, Mutability),
    // FIXME: Unbox this once `Idx` has a niche,
    // as `RefType` should shrink by 4 bytes then
    Reference(Box<RefType>),
    Array(ArrayType),
    Slice(TypeRefId),
    /// A fn pointer. Last element of the vector is the return type.
    Fn(Box<FnType>),
    ImplTrait(ThinVec<TypeBound>),
    DynTrait(ThinVec<TypeBound>),
    TypeParam(TypeParamId),
    Error,
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
const _: () = assert!(size_of::<TypeRef>() == 24);

pub type TypeRefId = Idx<TypeRef>;

pub type LifetimeRefId = Idx<LifetimeRef>;

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum LifetimeRef {
    Named(Name),
    Static,
    Placeholder,
    Param(LifetimeParamId),
    Error,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum TypeBound {
    Path(PathId, TraitBoundModifier),
    ForLifetime(ThinVec<Name>, PathId),
    Lifetime(LifetimeRefId),
    Use(ThinVec<UseArgRef>),
    Error,
}

#[cfg(target_pointer_width = "64")]
const _: [(); 16] = [(); size_of::<TypeBound>()];

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum UseArgRef {
    Name(Name),
    Lifetime(LifetimeRefId),
}

/// A modifier on a bound, currently this is only used for `?Sized`, where the
/// modifier is `Maybe`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TraitBoundModifier {
    None,
    Maybe,
}

impl TypeRef {
    pub(crate) fn unit() -> TypeRef {
        TypeRef::Tuple(ThinVec::new())
    }

    pub fn walk(this: TypeRefId, map: &ExpressionStore, f: &mut impl FnMut(&TypeRef)) {
        go(this, f, map);

        fn go(type_ref: TypeRefId, f: &mut impl FnMut(&TypeRef), map: &ExpressionStore) {
            let type_ref = &map[type_ref];
            f(type_ref);
            match type_ref {
                TypeRef::Fn(fn_) => {
                    fn_.params.iter().for_each(|&(_, param_type)| go(param_type, f, map))
                }
                TypeRef::Tuple(types) => types.iter().for_each(|&t| go(t, f, map)),
                TypeRef::RawPtr(type_ref, _) | TypeRef::Slice(type_ref) => go(*type_ref, f, map),
                TypeRef::Reference(it) => go(it.ty, f, map),
                TypeRef::Array(it) => go(it.ty, f, map),
                TypeRef::ImplTrait(bounds) | TypeRef::DynTrait(bounds) => {
                    for bound in bounds {
                        match bound {
                            &TypeBound::Path(path, _) | &TypeBound::ForLifetime(_, path) => {
                                go_path(&map[path], f, map)
                            }
                            TypeBound::Lifetime(_) | TypeBound::Error | TypeBound::Use(_) => (),
                        }
                    }
                }
                TypeRef::Path(path) => go_path(path, f, map),
                TypeRef::Never | TypeRef::Placeholder | TypeRef::Error | TypeRef::TypeParam(_) => {}
            };
        }

        fn go_path(path: &Path, f: &mut impl FnMut(&TypeRef), map: &ExpressionStore) {
            if let Some(type_ref) = path.type_anchor() {
                go(type_ref, f, map);
            }
            for segment in path.segments().iter() {
                if let Some(args_and_bindings) = segment.args_and_bindings {
                    for arg in args_and_bindings.args.iter() {
                        match arg {
                            GenericArg::Type(type_ref) => {
                                go(*type_ref, f, map);
                            }
                            GenericArg::Const(_) | GenericArg::Lifetime(_) => {}
                        }
                    }
                    for binding in args_and_bindings.bindings.iter() {
                        if let Some(type_ref) = binding.type_ref {
                            go(type_ref, f, map);
                        }
                        for bound in binding.bounds.iter() {
                            match bound {
                                &TypeBound::Path(path, _) | &TypeBound::ForLifetime(_, path) => {
                                    go_path(&map[path], f, map)
                                }
                                TypeBound::Lifetime(_) | TypeBound::Error | TypeBound::Use(_) => (),
                            }
                        }
                    }
                }
            }
        }
    }
}

impl TypeBound {
    pub fn as_path<'a>(&self, map: &'a ExpressionStore) -> Option<(&'a Path, TraitBoundModifier)> {
        match self {
            &TypeBound::Path(p, m) => Some((&map[p], m)),
            &TypeBound::ForLifetime(_, p) => Some((&map[p], TraitBoundModifier::None)),
            TypeBound::Lifetime(_) | TypeBound::Error | TypeBound::Use(_) => None,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ConstRef {
    pub expr: ExprId,
}

/// A literal constant value
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LiteralConstRef {
    Int(i128),
    UInt(u128),
    Bool(bool),
    Char(char),

    /// Case of an unknown value that rustc might know but we don't
    // FIXME: this is a hack to get around chalk not being able to represent unevaluatable
    // constants
    // https://github.com/rust-lang/rust-analyzer/pull/8813#issuecomment-840679177
    // https://rust-lang.zulipchat.com/#narrow/stream/144729-wg-traits/topic/Handling.20non.20evaluatable.20constants'.20equality/near/238386348
    Unknown,
}

impl LiteralConstRef {
    pub fn builtin_type(&self) -> BuiltinType {
        match self {
            LiteralConstRef::UInt(_) | LiteralConstRef::Unknown => {
                BuiltinType::Uint(BuiltinUint::U128)
            }
            LiteralConstRef::Int(_) => BuiltinType::Int(BuiltinInt::I128),
            LiteralConstRef::Char(_) => BuiltinType::Char,
            LiteralConstRef::Bool(_) => BuiltinType::Bool,
        }
    }
}

impl From<Literal> for LiteralConstRef {
    fn from(literal: Literal) -> Self {
        match literal {
            Literal::Char(c) => Self::Char(c),
            Literal::Bool(flag) => Self::Bool(flag),
            Literal::Int(num, _) => Self::Int(num),
            Literal::Uint(num, _) => Self::UInt(num),
            _ => Self::Unknown,
        }
    }
}

impl std::fmt::Display for LiteralConstRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            LiteralConstRef::Int(num) => num.fmt(f),
            LiteralConstRef::UInt(num) => num.fmt(f),
            LiteralConstRef::Bool(flag) => flag.fmt(f),
            LiteralConstRef::Char(c) => write!(f, "'{c}'"),
            LiteralConstRef::Unknown => f.write_char('_'),
        }
    }
}
