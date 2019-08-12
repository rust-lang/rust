//! The type system. We currently use this to infer types for completion, hover
//! information and various assists.

mod autoderef;
pub(crate) mod primitive;
#[cfg(test)]
mod tests;
pub(crate) mod traits;
pub(crate) mod method_resolution;
mod op;
mod lower;
mod infer;
pub(crate) mod display;

use std::ops::Deref;
use std::sync::Arc;
use std::{fmt, mem};

use crate::{db::HirDatabase, type_ref::Mutability, AdtDef, GenericParams, Name, Trait, TypeAlias};
use display::{HirDisplay, HirFormatter};

pub(crate) use autoderef::autoderef;
pub(crate) use infer::{infer_query, InferTy, InferenceResult};
pub use lower::CallableDef;
pub(crate) use lower::{
    callable_item_sig, generic_defaults_query, generic_predicates_query, type_for_def,
    type_for_field, TypableDef,
};
pub(crate) use traits::{InEnvironment, Obligation, ProjectionPredicate, TraitEnvironment};

/// A type constructor or type name: this might be something like the primitive
/// type `bool`, a struct like `Vec`, or things like function pointers or
/// tuples.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub enum TypeCtor {
    /// The primitive boolean type. Written as `bool`.
    Bool,

    /// The primitive character type; holds a Unicode scalar value
    /// (a non-surrogate code point). Written as `char`.
    Char,

    /// A primitive integer type. For example, `i32`.
    Int(primitive::UncertainIntTy),

    /// A primitive floating-point type. For example, `f64`.
    Float(primitive::UncertainFloatTy),

    /// Structures, enumerations and unions.
    Adt(AdtDef),

    /// The pointee of a string slice. Written as `str`.
    Str,

    /// The pointee of an array slice.  Written as `[T]`.
    Slice,

    /// An array with the given length. Written as `[T; n]`.
    Array,

    /// A raw pointer. Written as `*mut T` or `*const T`
    RawPtr(Mutability),

    /// A reference; a pointer with an associated lifetime. Written as
    /// `&'a mut T` or `&'a T`.
    Ref(Mutability),

    /// The anonymous type of a function declaration/definition. Each
    /// function has a unique type, which is output (for a function
    /// named `foo` returning an `i32`) as `fn() -> i32 {foo}`.
    ///
    /// This includes tuple struct / enum variant constructors as well.
    ///
    /// For example the type of `bar` here:
    ///
    /// ```rust
    /// fn foo() -> i32 { 1 }
    /// let bar = foo; // bar: fn() -> i32 {foo}
    /// ```
    FnDef(CallableDef),

    /// A pointer to a function.  Written as `fn() -> i32`.
    ///
    /// For example the type of `bar` here:
    ///
    /// ```rust
    /// fn foo() -> i32 { 1 }
    /// let bar: fn() -> i32 = foo;
    /// ```
    FnPtr { num_args: u16 },

    /// The never type `!`.
    Never,

    /// A tuple type.  For example, `(i32, bool)`.
    Tuple { cardinality: u16 },

    /// Represents an associated item like `Iterator::Item`.  This is used
    /// when we have tried to normalize a projection like `T::Item` but
    /// couldn't find a better representation.  In that case, we generate
    /// an **application type** like `(Iterator::Item)<T>`.
    AssociatedType(TypeAlias),
}

/// A nominal type with (maybe 0) type parameters. This might be a primitive
/// type like `bool`, a struct, tuple, function pointer, reference or
/// several other things.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct ApplicationTy {
    pub ctor: TypeCtor,
    pub parameters: Substs,
}

/// A "projection" type corresponds to an (unnormalized)
/// projection like `<P0 as Trait<P1..Pn>>::Foo`. Note that the
/// trait and all its parameters are fully known.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct ProjectionTy {
    pub associated_ty: TypeAlias,
    pub parameters: Substs,
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct UnselectedProjectionTy {
    pub type_name: Name,
    pub parameters: Substs,
}

/// A type.
///
/// See also the `TyKind` enum in rustc (librustc/ty/sty.rs), which represents
/// the same thing (but in a different way).
///
/// This should be cheap to clone.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub enum Ty {
    /// A nominal type with (maybe 0) type parameters. This might be a primitive
    /// type like `bool`, a struct, tuple, function pointer, reference or
    /// several other things.
    Apply(ApplicationTy),

    /// A "projection" type corresponds to an (unnormalized)
    /// projection like `<P0 as Trait<P1..Pn>>::Foo`. Note that the
    /// trait and all its parameters are fully known.
    Projection(ProjectionTy),

    /// This is a variant of a projection in which the trait is
    /// **not** known.  It corresponds to a case where people write
    /// `T::Item` without specifying the trait. We would then try to
    /// figure out the trait by looking at all the traits that are in
    /// scope.
    UnselectedProjection(UnselectedProjectionTy),

    /// A type parameter; for example, `T` in `fn f<T>(x: T) {}
    Param {
        /// The index of the parameter (starting with parameters from the
        /// surrounding impl, then the current function).
        idx: u32,
        /// The name of the parameter, for displaying.
        // FIXME get rid of this
        name: Name,
    },

    /// A bound type variable. Only used during trait resolution to represent
    /// Chalk variables.
    Bound(u32),

    /// A type variable used during type checking. Not to be confused with a
    /// type parameter.
    Infer(InferTy),

    /// A placeholder for a type which could not be computed; this is propagated
    /// to avoid useless error messages. Doubles as a placeholder where type
    /// variables are inserted before type checking, since we want to try to
    /// infer a better type here anyway -- for the IDE use case, we want to try
    /// to infer as much as possible even in the presence of type errors.
    Unknown,
}

/// A list of substitutions for generic parameters.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct Substs(Arc<[Ty]>);

impl Substs {
    pub fn empty() -> Substs {
        Substs(Arc::new([]))
    }

    pub fn single(ty: Ty) -> Substs {
        Substs(Arc::new([ty]))
    }

    pub fn prefix(&self, n: usize) -> Substs {
        Substs(self.0.iter().cloned().take(n).collect::<Vec<_>>().into())
    }

    pub fn walk_mut(&mut self, f: &mut impl FnMut(&mut Ty)) {
        // Without an Arc::make_mut_slice, we can't avoid the clone here:
        let mut v: Vec<_> = self.0.iter().cloned().collect();
        for t in &mut v {
            t.walk_mut(f);
        }
        self.0 = v.into();
    }

    pub fn as_single(&self) -> &Ty {
        if self.0.len() != 1 {
            panic!("expected substs of len 1, got {:?}", self);
        }
        &self.0[0]
    }

    /// Return Substs that replace each parameter by itself (i.e. `Ty::Param`).
    pub fn identity(generic_params: &GenericParams) -> Substs {
        Substs(
            generic_params
                .params_including_parent()
                .into_iter()
                .map(|p| Ty::Param { idx: p.idx, name: p.name.clone() })
                .collect::<Vec<_>>()
                .into(),
        )
    }

    /// Return Substs that replace each parameter by a bound variable.
    pub fn bound_vars(generic_params: &GenericParams) -> Substs {
        Substs(
            generic_params
                .params_including_parent()
                .into_iter()
                .map(|p| Ty::Bound(p.idx))
                .collect::<Vec<_>>()
                .into(),
        )
    }
}

impl From<Vec<Ty>> for Substs {
    fn from(v: Vec<Ty>) -> Self {
        Substs(v.into())
    }
}

impl Deref for Substs {
    type Target = [Ty];

    fn deref(&self) -> &[Ty] {
        &self.0
    }
}

/// A trait with type parameters. This includes the `Self`, so this represents a concrete type implementing the trait.
/// Name to be bikeshedded: TraitBound? TraitImplements?
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct TraitRef {
    /// FIXME name?
    pub trait_: Trait,
    pub substs: Substs,
}

impl TraitRef {
    pub fn self_ty(&self) -> &Ty {
        &self.substs[0]
    }

    pub fn subst(mut self, substs: &Substs) -> TraitRef {
        self.substs.walk_mut(&mut |ty_mut| {
            let ty = mem::replace(ty_mut, Ty::Unknown);
            *ty_mut = ty.subst(substs);
        });
        self
    }
}

/// Like `generics::WherePredicate`, but with resolved types: A condition on the
/// parameters of a generic item.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GenericPredicate {
    /// The given trait needs to be implemented for its type parameters.
    Implemented(TraitRef),
    /// We couldn't resolve the trait reference. (If some type parameters can't
    /// be resolved, they will just be Unknown).
    Error,
}

impl GenericPredicate {
    pub fn is_error(&self) -> bool {
        match self {
            GenericPredicate::Error => true,
            _ => false,
        }
    }

    pub fn subst(self, substs: &Substs) -> GenericPredicate {
        match self {
            GenericPredicate::Implemented(trait_ref) => {
                GenericPredicate::Implemented(trait_ref.subst(substs))
            }
            GenericPredicate::Error => self,
        }
    }
}

/// Basically a claim (currently not validated / checked) that the contained
/// type / trait ref contains no inference variables; any inference variables it
/// contained have been replaced by bound variables, and `num_vars` tells us how
/// many there are. This is used to erase irrelevant differences between types
/// before using them in queries.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Canonical<T> {
    pub value: T,
    pub num_vars: usize,
}

/// A function signature as seen by type inference: Several parameter types and
/// one return type.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FnSig {
    params_and_return: Arc<[Ty]>,
}

impl FnSig {
    pub fn from_params_and_return(mut params: Vec<Ty>, ret: Ty) -> FnSig {
        params.push(ret);
        FnSig { params_and_return: params.into() }
    }

    pub fn from_fn_ptr_substs(substs: &Substs) -> FnSig {
        FnSig { params_and_return: Arc::clone(&substs.0) }
    }

    pub fn params(&self) -> &[Ty] {
        &self.params_and_return[0..self.params_and_return.len() - 1]
    }

    pub fn ret(&self) -> &Ty {
        &self.params_and_return[self.params_and_return.len() - 1]
    }

    /// Applies the given substitutions to all types in this signature and
    /// returns the result.
    pub fn subst(&self, substs: &Substs) -> FnSig {
        let result: Vec<_> =
            self.params_and_return.iter().map(|ty| ty.clone().subst(substs)).collect();
        FnSig { params_and_return: result.into() }
    }

    pub fn walk_mut(&mut self, f: &mut impl FnMut(&mut Ty)) {
        // Without an Arc::make_mut_slice, we can't avoid the clone here:
        let mut v: Vec<_> = self.params_and_return.iter().cloned().collect();
        for t in &mut v {
            t.walk_mut(f);
        }
        self.params_and_return = v.into();
    }
}

impl Ty {
    pub fn simple(ctor: TypeCtor) -> Ty {
        Ty::Apply(ApplicationTy { ctor, parameters: Substs::empty() })
    }
    pub fn apply_one(ctor: TypeCtor, param: Ty) -> Ty {
        Ty::Apply(ApplicationTy { ctor, parameters: Substs::single(param) })
    }
    pub fn apply(ctor: TypeCtor, parameters: Substs) -> Ty {
        Ty::Apply(ApplicationTy { ctor, parameters })
    }
    pub fn unit() -> Self {
        Ty::apply(TypeCtor::Tuple { cardinality: 0 }, Substs::empty())
    }

    pub fn walk(&self, f: &mut impl FnMut(&Ty)) {
        match self {
            Ty::Apply(a_ty) => {
                for t in a_ty.parameters.iter() {
                    t.walk(f);
                }
            }
            Ty::Projection(p_ty) => {
                for t in p_ty.parameters.iter() {
                    t.walk(f);
                }
            }
            Ty::UnselectedProjection(p_ty) => {
                for t in p_ty.parameters.iter() {
                    t.walk(f);
                }
            }
            Ty::Param { .. } | Ty::Bound(_) | Ty::Infer(_) | Ty::Unknown => {}
        }
        f(self);
    }

    fn walk_mut(&mut self, f: &mut impl FnMut(&mut Ty)) {
        match self {
            Ty::Apply(a_ty) => {
                a_ty.parameters.walk_mut(f);
            }
            Ty::Projection(p_ty) => {
                p_ty.parameters.walk_mut(f);
            }
            Ty::UnselectedProjection(p_ty) => {
                p_ty.parameters.walk_mut(f);
            }
            Ty::Param { .. } | Ty::Bound(_) | Ty::Infer(_) | Ty::Unknown => {}
        }
        f(self);
    }

    fn fold(mut self, f: &mut impl FnMut(Ty) -> Ty) -> Ty {
        self.walk_mut(&mut |ty_mut| {
            let ty = mem::replace(ty_mut, Ty::Unknown);
            *ty_mut = f(ty);
        });
        self
    }

    pub fn as_reference(&self) -> Option<(&Ty, Mutability)> {
        match self {
            Ty::Apply(ApplicationTy { ctor: TypeCtor::Ref(mutability), parameters }) => {
                Some((parameters.as_single(), *mutability))
            }
            _ => None,
        }
    }

    pub fn as_adt(&self) -> Option<(AdtDef, &Substs)> {
        match self {
            Ty::Apply(ApplicationTy { ctor: TypeCtor::Adt(adt_def), parameters }) => {
                Some((*adt_def, parameters))
            }
            _ => None,
        }
    }

    pub fn as_tuple(&self) -> Option<&Substs> {
        match self {
            Ty::Apply(ApplicationTy { ctor: TypeCtor::Tuple { .. }, parameters }) => {
                Some(parameters)
            }
            _ => None,
        }
    }

    pub fn as_callable(&self) -> Option<(CallableDef, &Substs)> {
        match self {
            Ty::Apply(ApplicationTy { ctor: TypeCtor::FnDef(callable_def), parameters }) => {
                Some((*callable_def, parameters))
            }
            _ => None,
        }
    }

    fn builtin_deref(&self) -> Option<Ty> {
        match self {
            Ty::Apply(a_ty) => match a_ty.ctor {
                TypeCtor::Ref(..) => Some(Ty::clone(a_ty.parameters.as_single())),
                TypeCtor::RawPtr(..) => Some(Ty::clone(a_ty.parameters.as_single())),
                _ => None,
            },
            _ => None,
        }
    }

    fn callable_sig(&self, db: &impl HirDatabase) -> Option<FnSig> {
        match self {
            Ty::Apply(a_ty) => match a_ty.ctor {
                TypeCtor::FnPtr { .. } => Some(FnSig::from_fn_ptr_substs(&a_ty.parameters)),
                TypeCtor::FnDef(def) => {
                    let sig = db.callable_item_signature(def);
                    Some(sig.subst(&a_ty.parameters))
                }
                _ => None,
            },
            _ => None,
        }
    }

    /// If this is a type with type parameters (an ADT or function), replaces
    /// the `Substs` for these type parameters with the given ones. (So e.g. if
    /// `self` is `Option<_>` and the substs contain `u32`, we'll have
    /// `Option<u32>` afterwards.)
    pub fn apply_substs(self, substs: Substs) -> Ty {
        match self {
            Ty::Apply(ApplicationTy { ctor, parameters: previous_substs }) => {
                assert_eq!(previous_substs.len(), substs.len());
                Ty::Apply(ApplicationTy { ctor, parameters: substs })
            }
            _ => self,
        }
    }

    /// Replaces type parameters in this type using the given `Substs`. (So e.g.
    /// if `self` is `&[T]`, where type parameter T has index 0, and the
    /// `Substs` contain `u32` at index 0, we'll have `&[u32]` afterwards.)
    pub fn subst(self, substs: &Substs) -> Ty {
        self.fold(&mut |ty| match ty {
            Ty::Param { idx, name } => {
                substs.get(idx as usize).cloned().unwrap_or(Ty::Param { idx, name })
            }
            ty => ty,
        })
    }

    /// Substitutes `Ty::Bound` vars (as opposed to type parameters).
    pub fn subst_bound_vars(self, substs: &Substs) -> Ty {
        self.fold(&mut |ty| match ty {
            Ty::Bound(idx) => substs.get(idx as usize).cloned().unwrap_or_else(|| Ty::Bound(idx)),
            ty => ty,
        })
    }

    /// Returns the type parameters of this type if it has some (i.e. is an ADT
    /// or function); so if `self` is `Option<u32>`, this returns the `u32`.
    pub fn substs(&self) -> Option<Substs> {
        match self {
            Ty::Apply(ApplicationTy { parameters, .. }) => Some(parameters.clone()),
            _ => None,
        }
    }

    /// Shifts up `Ty::Bound` vars by `n`.
    pub fn shift_bound_vars(self, n: i32) -> Ty {
        self.fold(&mut |ty| match ty {
            Ty::Bound(idx) => {
                assert!(idx as i32 >= -n);
                Ty::Bound((idx as i32 + n) as u32)
            }
            ty => ty,
        })
    }
}

impl HirDisplay for &Ty {
    fn hir_fmt(&self, f: &mut HirFormatter<impl HirDatabase>) -> fmt::Result {
        HirDisplay::hir_fmt(*self, f)
    }
}

impl HirDisplay for ApplicationTy {
    fn hir_fmt(&self, f: &mut HirFormatter<impl HirDatabase>) -> fmt::Result {
        match self.ctor {
            TypeCtor::Bool => write!(f, "bool")?,
            TypeCtor::Char => write!(f, "char")?,
            TypeCtor::Int(t) => write!(f, "{}", t)?,
            TypeCtor::Float(t) => write!(f, "{}", t)?,
            TypeCtor::Str => write!(f, "str")?,
            TypeCtor::Slice => {
                let t = self.parameters.as_single();
                write!(f, "[{}]", t.display(f.db))?;
            }
            TypeCtor::Array => {
                let t = self.parameters.as_single();
                write!(f, "[{};_]", t.display(f.db))?;
            }
            TypeCtor::RawPtr(m) => {
                let t = self.parameters.as_single();
                write!(f, "*{}{}", m.as_keyword_for_ptr(), t.display(f.db))?;
            }
            TypeCtor::Ref(m) => {
                let t = self.parameters.as_single();
                write!(f, "&{}{}", m.as_keyword_for_ref(), t.display(f.db))?;
            }
            TypeCtor::Never => write!(f, "!")?,
            TypeCtor::Tuple { .. } => {
                let ts = &self.parameters;
                if ts.len() == 1 {
                    write!(f, "({},)", ts[0].display(f.db))?;
                } else {
                    write!(f, "(")?;
                    f.write_joined(&*ts.0, ", ")?;
                    write!(f, ")")?;
                }
            }
            TypeCtor::FnPtr { .. } => {
                let sig = FnSig::from_fn_ptr_substs(&self.parameters);
                write!(f, "fn(")?;
                f.write_joined(sig.params(), ", ")?;
                write!(f, ") -> {}", sig.ret().display(f.db))?;
            }
            TypeCtor::FnDef(def) => {
                let sig = f.db.callable_item_signature(def);
                let name = match def {
                    CallableDef::Function(ff) => ff.name(f.db),
                    CallableDef::Struct(s) => s.name(f.db).unwrap_or_else(Name::missing),
                    CallableDef::EnumVariant(e) => e.name(f.db).unwrap_or_else(Name::missing),
                };
                match def {
                    CallableDef::Function(_) => write!(f, "fn {}", name)?,
                    CallableDef::Struct(_) | CallableDef::EnumVariant(_) => write!(f, "{}", name)?,
                }
                if self.parameters.len() > 0 {
                    write!(f, "<")?;
                    f.write_joined(&*self.parameters.0, ", ")?;
                    write!(f, ">")?;
                }
                write!(f, "(")?;
                f.write_joined(sig.params(), ", ")?;
                write!(f, ") -> {}", sig.ret().display(f.db))?;
            }
            TypeCtor::Adt(def_id) => {
                let name = match def_id {
                    AdtDef::Struct(s) => s.name(f.db),
                    AdtDef::Union(u) => u.name(f.db),
                    AdtDef::Enum(e) => e.name(f.db),
                }
                .unwrap_or_else(Name::missing);
                write!(f, "{}", name)?;
                if self.parameters.len() > 0 {
                    write!(f, "<")?;
                    f.write_joined(&*self.parameters.0, ", ")?;
                    write!(f, ">")?;
                }
            }
            TypeCtor::AssociatedType(type_alias) => {
                let trait_name = type_alias
                    .parent_trait(f.db)
                    .and_then(|t| t.name(f.db))
                    .unwrap_or_else(Name::missing);
                let name = type_alias.name(f.db);
                write!(f, "{}::{}", trait_name, name)?;
                if self.parameters.len() > 0 {
                    write!(f, "<")?;
                    f.write_joined(&*self.parameters.0, ", ")?;
                    write!(f, ">")?;
                }
            }
        }
        Ok(())
    }
}

impl HirDisplay for ProjectionTy {
    fn hir_fmt(&self, f: &mut HirFormatter<impl HirDatabase>) -> fmt::Result {
        let trait_name = self
            .associated_ty
            .parent_trait(f.db)
            .and_then(|t| t.name(f.db))
            .unwrap_or_else(Name::missing);
        write!(f, "<{} as {}", self.parameters[0].display(f.db), trait_name,)?;
        if self.parameters.len() > 1 {
            write!(f, "<")?;
            f.write_joined(&self.parameters[1..], ", ")?;
            write!(f, ">")?;
        }
        write!(f, ">::{}", self.associated_ty.name(f.db))?;
        Ok(())
    }
}

impl HirDisplay for UnselectedProjectionTy {
    fn hir_fmt(&self, f: &mut HirFormatter<impl HirDatabase>) -> fmt::Result {
        write!(f, "{}", self.parameters[0].display(f.db))?;
        if self.parameters.len() > 1 {
            write!(f, "<")?;
            f.write_joined(&self.parameters[1..], ", ")?;
            write!(f, ">")?;
        }
        write!(f, "::{}", self.type_name)?;
        Ok(())
    }
}

impl HirDisplay for Ty {
    fn hir_fmt(&self, f: &mut HirFormatter<impl HirDatabase>) -> fmt::Result {
        match self {
            Ty::Apply(a_ty) => a_ty.hir_fmt(f)?,
            Ty::Projection(p_ty) => p_ty.hir_fmt(f)?,
            Ty::UnselectedProjection(p_ty) => p_ty.hir_fmt(f)?,
            Ty::Param { name, .. } => write!(f, "{}", name)?,
            Ty::Bound(idx) => write!(f, "?{}", idx)?,
            Ty::Unknown => write!(f, "{{unknown}}")?,
            Ty::Infer(..) => write!(f, "_")?,
        }
        Ok(())
    }
}

impl HirDisplay for TraitRef {
    fn hir_fmt(&self, f: &mut HirFormatter<impl HirDatabase>) -> fmt::Result {
        write!(
            f,
            "{}: {}",
            self.substs[0].display(f.db),
            self.trait_.name(f.db).unwrap_or_else(Name::missing)
        )?;
        if self.substs.len() > 1 {
            write!(f, "<")?;
            f.write_joined(&self.substs[1..], ", ")?;
            write!(f, ">")?;
        }
        Ok(())
    }
}

impl HirDisplay for Obligation {
    fn hir_fmt(&self, f: &mut HirFormatter<impl HirDatabase>) -> fmt::Result {
        match self {
            Obligation::Trait(tr) => write!(f, "Implements({})", tr.display(f.db)),
            Obligation::Projection(proj) => write!(
                f,
                "Normalize({} => {})",
                proj.projection_ty.display(f.db),
                proj.ty.display(f.db)
            ),
        }
    }
}
