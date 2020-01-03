//! The type system. We currently use this to infer types for completion, hover
//! information and various assists.

macro_rules! impl_froms {
    ($e:ident: $($v:ident $(($($sv:ident),*))?),*) => {
        $(
            impl From<$v> for $e {
                fn from(it: $v) -> $e {
                    $e::$v(it)
                }
            }
            $($(
                impl From<$sv> for $e {
                    fn from(it: $sv) -> $e {
                        $e::$v($v::$sv(it))
                    }
                }
            )*)?
        )*
    }
}

mod autoderef;
pub mod primitive;
pub mod traits;
pub mod method_resolution;
mod op;
mod lower;
mod infer;
pub mod display;
pub(crate) mod utils;
pub mod db;
pub mod diagnostics;
pub mod expr;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod test_db;
mod marks;

use std::ops::Deref;
use std::sync::Arc;
use std::{fmt, iter, mem};

use hir_def::{
    expr::ExprId, type_ref::Mutability, AdtId, AssocContainerId, DefWithBodyId, GenericDefId,
    HasModule, Lookup, TraitId, TypeAliasId,
};
use hir_expand::name::Name;
use ra_db::{impl_intern_key, salsa, CrateId};

use crate::{
    db::HirDatabase,
    primitive::{FloatTy, IntTy, Uncertain},
    utils::{generics, make_mut_slice, Generics},
};
use display::{HirDisplay, HirFormatter};

pub use autoderef::autoderef;
pub use infer::{do_infer_query, InferTy, InferenceResult};
pub use lower::CallableDef;
pub use lower::{callable_item_sig, TyDefId, ValueTyDefId};
pub use traits::{InEnvironment, Obligation, ProjectionPredicate, TraitEnvironment};

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
    Int(Uncertain<IntTy>),

    /// A primitive floating-point type. For example, `f64`.
    Float(Uncertain<FloatTy>),

    /// Structures, enumerations and unions.
    Adt(AdtId),

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
    /// ```
    /// fn foo() -> i32 { 1 }
    /// let bar = foo; // bar: fn() -> i32 {foo}
    /// ```
    FnDef(CallableDef),

    /// A pointer to a function.  Written as `fn() -> i32`.
    ///
    /// For example the type of `bar` here:
    ///
    /// ```
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
    AssociatedType(TypeAliasId),

    /// The type of a specific closure.
    ///
    /// The closure signature is stored in a `FnPtr` type in the first type
    /// parameter.
    Closure { def: DefWithBodyId, expr: ExprId },
}

/// This exists just for Chalk, because Chalk just has a single `StructId` where
/// we have different kinds of ADTs, primitive types and special type
/// constructors like tuples and function pointers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeCtorId(salsa::InternId);
impl_intern_key!(TypeCtorId);

impl TypeCtor {
    pub fn num_ty_params(self, db: &impl HirDatabase) -> usize {
        match self {
            TypeCtor::Bool
            | TypeCtor::Char
            | TypeCtor::Int(_)
            | TypeCtor::Float(_)
            | TypeCtor::Str
            | TypeCtor::Never => 0,
            TypeCtor::Slice
            | TypeCtor::Array
            | TypeCtor::RawPtr(_)
            | TypeCtor::Ref(_)
            | TypeCtor::Closure { .. } // 1 param representing the signature of the closure
            => 1,
            TypeCtor::Adt(adt) => {
                let generic_params = generics(db, AdtId::from(adt).into());
                generic_params.len()
            }
            TypeCtor::FnDef(callable) => {
                let generic_params = generics(db, callable.into());
                generic_params.len()
            }
            TypeCtor::AssociatedType(type_alias) => {
                let generic_params = generics(db, type_alias.into());
                generic_params.len()
            }
            TypeCtor::FnPtr { num_args } => num_args as usize + 1,
            TypeCtor::Tuple { cardinality } => cardinality as usize,
        }
    }

    pub fn krate(self, db: &impl HirDatabase) -> Option<CrateId> {
        match self {
            TypeCtor::Bool
            | TypeCtor::Char
            | TypeCtor::Int(_)
            | TypeCtor::Float(_)
            | TypeCtor::Str
            | TypeCtor::Never
            | TypeCtor::Slice
            | TypeCtor::Array
            | TypeCtor::RawPtr(_)
            | TypeCtor::Ref(_)
            | TypeCtor::FnPtr { .. }
            | TypeCtor::Tuple { .. } => None,
            // Closure's krate is irrelevant for coherence I would think?
            TypeCtor::Closure { .. } => None,
            TypeCtor::Adt(adt) => Some(adt.module(db).krate),
            TypeCtor::FnDef(callable) => Some(callable.krate(db)),
            TypeCtor::AssociatedType(type_alias) => Some(type_alias.lookup(db).module(db).krate),
        }
    }

    pub fn as_generic_def(self) -> Option<GenericDefId> {
        match self {
            TypeCtor::Bool
            | TypeCtor::Char
            | TypeCtor::Int(_)
            | TypeCtor::Float(_)
            | TypeCtor::Str
            | TypeCtor::Never
            | TypeCtor::Slice
            | TypeCtor::Array
            | TypeCtor::RawPtr(_)
            | TypeCtor::Ref(_)
            | TypeCtor::FnPtr { .. }
            | TypeCtor::Tuple { .. }
            | TypeCtor::Closure { .. } => None,
            TypeCtor::Adt(adt) => Some(adt.into()),
            TypeCtor::FnDef(callable) => Some(callable.into()),
            TypeCtor::AssociatedType(type_alias) => Some(type_alias.into()),
        }
    }
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
    pub associated_ty: TypeAliasId,
    pub parameters: Substs,
}

impl ProjectionTy {
    pub fn trait_ref(&self, db: &impl HirDatabase) -> TraitRef {
        TraitRef { trait_: self.trait_(db).into(), substs: self.parameters.clone() }
    }

    fn trait_(&self, db: &impl HirDatabase) -> TraitId {
        match self.associated_ty.lookup(db).container {
            AssocContainerId::TraitId(it) => it,
            _ => panic!("projection ty without parent trait"),
        }
    }
}

impl TypeWalk for ProjectionTy {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        self.parameters.walk(f);
    }

    fn walk_mut_binders(&mut self, f: &mut impl FnMut(&mut Ty, usize), binders: usize) {
        self.parameters.walk_mut_binders(f, binders);
    }
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

    /// A type parameter; for example, `T` in `fn f<T>(x: T) {}
    Param {
        /// The index of the parameter (starting with parameters from the
        /// surrounding impl, then the current function).
        idx: u32,
        /// The name of the parameter, for displaying.
        // FIXME get rid of this
        name: Name,
    },

    /// A bound type variable. Used during trait resolution to represent Chalk
    /// variables, and in `Dyn` and `Opaque` bounds to represent the `Self` type.
    Bound(u32),

    /// A type variable used during type checking. Not to be confused with a
    /// type parameter.
    Infer(InferTy),

    /// A trait object (`dyn Trait` or bare `Trait` in pre-2018 Rust).
    ///
    /// The predicates are quantified over the `Self` type, i.e. `Ty::Bound(0)`
    /// represents the `Self` type inside the bounds. This is currently
    /// implicit; Chalk has the `Binders` struct to make it explicit, but it
    /// didn't seem worth the overhead yet.
    Dyn(Arc<[GenericPredicate]>),

    /// An opaque type (`impl Trait`).
    ///
    /// The predicates are quantified over the `Self` type; see `Ty::Dyn` for
    /// more.
    Opaque(Arc<[GenericPredicate]>),

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

impl TypeWalk for Substs {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        for t in self.0.iter() {
            t.walk(f);
        }
    }

    fn walk_mut_binders(&mut self, f: &mut impl FnMut(&mut Ty, usize), binders: usize) {
        for t in make_mut_slice(&mut self.0) {
            t.walk_mut_binders(f, binders);
        }
    }
}

impl Substs {
    pub fn empty() -> Substs {
        Substs(Arc::new([]))
    }

    pub fn single(ty: Ty) -> Substs {
        Substs(Arc::new([ty]))
    }

    pub fn prefix(&self, n: usize) -> Substs {
        Substs(self.0[..std::cmp::min(self.0.len(), n)].into())
    }

    pub fn as_single(&self) -> &Ty {
        if self.0.len() != 1 {
            panic!("expected substs of len 1, got {:?}", self);
        }
        &self.0[0]
    }

    /// Return Substs that replace each parameter by itself (i.e. `Ty::Param`).
    pub(crate) fn identity(generic_params: &Generics) -> Substs {
        Substs(
            generic_params.iter().map(|(idx, p)| Ty::Param { idx, name: p.name.clone() }).collect(),
        )
    }

    /// Return Substs that replace each parameter by a bound variable.
    pub(crate) fn bound_vars(generic_params: &Generics) -> Substs {
        Substs(generic_params.iter().map(|(idx, _p)| Ty::Bound(idx)).collect())
    }

    pub fn build_for_def(db: &impl HirDatabase, def: impl Into<GenericDefId>) -> SubstsBuilder {
        let def = def.into();
        let params = generics(db, def);
        let param_count = params.len();
        Substs::builder(param_count)
    }

    pub(crate) fn build_for_generics(generic_params: &Generics) -> SubstsBuilder {
        Substs::builder(generic_params.len())
    }

    pub fn build_for_type_ctor(db: &impl HirDatabase, type_ctor: TypeCtor) -> SubstsBuilder {
        Substs::builder(type_ctor.num_ty_params(db))
    }

    fn builder(param_count: usize) -> SubstsBuilder {
        SubstsBuilder { vec: Vec::with_capacity(param_count), param_count }
    }
}

#[derive(Debug, Clone)]
pub struct SubstsBuilder {
    vec: Vec<Ty>,
    param_count: usize,
}

impl SubstsBuilder {
    pub fn build(self) -> Substs {
        assert_eq!(self.vec.len(), self.param_count);
        Substs(self.vec.into())
    }

    pub fn push(mut self, ty: Ty) -> Self {
        self.vec.push(ty);
        self
    }

    fn remaining(&self) -> usize {
        self.param_count - self.vec.len()
    }

    pub fn fill_with_bound_vars(self, starting_from: u32) -> Self {
        self.fill((starting_from..).map(Ty::Bound))
    }

    pub fn fill_with_params(self) -> Self {
        let start = self.vec.len() as u32;
        self.fill((start..).map(|idx| Ty::Param { idx, name: Name::missing() }))
    }

    pub fn fill_with_unknown(self) -> Self {
        self.fill(iter::repeat(Ty::Unknown))
    }

    pub fn fill(mut self, filler: impl Iterator<Item = Ty>) -> Self {
        self.vec.extend(filler.take(self.remaining()));
        assert_eq!(self.remaining(), 0);
        self
    }

    pub fn use_parent_substs(mut self, parent_substs: &Substs) -> Self {
        assert!(self.vec.is_empty());
        assert!(parent_substs.len() <= self.param_count);
        self.vec.extend(parent_substs.iter().cloned());
        self
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
    pub trait_: TraitId,
    pub substs: Substs,
}

impl TraitRef {
    pub fn self_ty(&self) -> &Ty {
        &self.substs[0]
    }
}

impl TypeWalk for TraitRef {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        self.substs.walk(f);
    }

    fn walk_mut_binders(&mut self, f: &mut impl FnMut(&mut Ty, usize), binders: usize) {
        self.substs.walk_mut_binders(f, binders);
    }
}

/// Like `generics::WherePredicate`, but with resolved types: A condition on the
/// parameters of a generic item.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GenericPredicate {
    /// The given trait needs to be implemented for its type parameters.
    Implemented(TraitRef),
    /// An associated type bindings like in `Iterator<Item = T>`.
    Projection(ProjectionPredicate),
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

    pub fn is_implemented(&self) -> bool {
        match self {
            GenericPredicate::Implemented(_) => true,
            _ => false,
        }
    }

    pub fn trait_ref(&self, db: &impl HirDatabase) -> Option<TraitRef> {
        match self {
            GenericPredicate::Implemented(tr) => Some(tr.clone()),
            GenericPredicate::Projection(proj) => Some(proj.projection_ty.trait_ref(db)),
            GenericPredicate::Error => None,
        }
    }
}

impl TypeWalk for GenericPredicate {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        match self {
            GenericPredicate::Implemented(trait_ref) => trait_ref.walk(f),
            GenericPredicate::Projection(projection_pred) => projection_pred.walk(f),
            GenericPredicate::Error => {}
        }
    }

    fn walk_mut_binders(&mut self, f: &mut impl FnMut(&mut Ty, usize), binders: usize) {
        match self {
            GenericPredicate::Implemented(trait_ref) => trait_ref.walk_mut_binders(f, binders),
            GenericPredicate::Projection(projection_pred) => {
                projection_pred.walk_mut_binders(f, binders)
            }
            GenericPredicate::Error => {}
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
}

impl TypeWalk for FnSig {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        for t in self.params_and_return.iter() {
            t.walk(f);
        }
    }

    fn walk_mut_binders(&mut self, f: &mut impl FnMut(&mut Ty, usize), binders: usize) {
        for t in make_mut_slice(&mut self.params_and_return) {
            t.walk_mut_binders(f, binders);
        }
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

    pub fn as_reference(&self) -> Option<(&Ty, Mutability)> {
        match self {
            Ty::Apply(ApplicationTy { ctor: TypeCtor::Ref(mutability), parameters }) => {
                Some((parameters.as_single(), *mutability))
            }
            _ => None,
        }
    }

    pub fn as_adt(&self) -> Option<(AdtId, &Substs)> {
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
                TypeCtor::Closure { .. } => {
                    let sig_param = &a_ty.parameters[0];
                    sig_param.callable_sig(db)
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

    /// Returns the type parameters of this type if it has some (i.e. is an ADT
    /// or function); so if `self` is `Option<u32>`, this returns the `u32`.
    pub fn substs(&self) -> Option<Substs> {
        match self {
            Ty::Apply(ApplicationTy { parameters, .. }) => Some(parameters.clone()),
            _ => None,
        }
    }

    /// If this is an `impl Trait` or `dyn Trait`, returns that trait.
    pub fn inherent_trait(&self) -> Option<TraitId> {
        match self {
            Ty::Dyn(predicates) | Ty::Opaque(predicates) => {
                predicates.iter().find_map(|pred| match pred {
                    GenericPredicate::Implemented(tr) => Some(tr.trait_),
                    _ => None,
                })
            }
            _ => None,
        }
    }
}

/// This allows walking structures that contain types to do something with those
/// types, similar to Chalk's `Fold` trait.
pub trait TypeWalk {
    fn walk(&self, f: &mut impl FnMut(&Ty));
    fn walk_mut(&mut self, f: &mut impl FnMut(&mut Ty)) {
        self.walk_mut_binders(&mut |ty, _binders| f(ty), 0);
    }
    /// Walk the type, counting entered binders.
    ///
    /// `Ty::Bound` variables use DeBruijn indexing, which means that 0 refers
    /// to the innermost binder, 1 to the next, etc.. So when we want to
    /// substitute a certain bound variable, we can't just walk the whole type
    /// and blindly replace each instance of a certain index; when we 'enter'
    /// things that introduce new bound variables, we have to keep track of
    /// that. Currently, the only thing that introduces bound variables on our
    /// side are `Ty::Dyn` and `Ty::Opaque`, which each introduce a bound
    /// variable for the self type.
    fn walk_mut_binders(&mut self, f: &mut impl FnMut(&mut Ty, usize), binders: usize);

    fn fold(mut self, f: &mut impl FnMut(Ty) -> Ty) -> Self
    where
        Self: Sized,
    {
        self.walk_mut(&mut |ty_mut| {
            let ty = mem::replace(ty_mut, Ty::Unknown);
            *ty_mut = f(ty);
        });
        self
    }

    /// Replaces type parameters in this type using the given `Substs`. (So e.g.
    /// if `self` is `&[T]`, where type parameter T has index 0, and the
    /// `Substs` contain `u32` at index 0, we'll have `&[u32]` afterwards.)
    fn subst(self, substs: &Substs) -> Self
    where
        Self: Sized,
    {
        self.fold(&mut |ty| match ty {
            Ty::Param { idx, name } => {
                substs.get(idx as usize).cloned().unwrap_or(Ty::Param { idx, name })
            }
            ty => ty,
        })
    }

    /// Substitutes `Ty::Bound` vars (as opposed to type parameters).
    fn subst_bound_vars(mut self, substs: &Substs) -> Self
    where
        Self: Sized,
    {
        self.walk_mut_binders(
            &mut |ty, binders| match ty {
                &mut Ty::Bound(idx) => {
                    if idx as usize >= binders && (idx as usize - binders) < substs.len() {
                        *ty = substs.0[idx as usize - binders].clone();
                    }
                }
                _ => {}
            },
            0,
        );
        self
    }

    /// Shifts up `Ty::Bound` vars by `n`.
    fn shift_bound_vars(self, n: i32) -> Self
    where
        Self: Sized,
    {
        self.fold(&mut |ty| match ty {
            Ty::Bound(idx) => {
                assert!(idx as i32 >= -n);
                Ty::Bound((idx as i32 + n) as u32)
            }
            ty => ty,
        })
    }
}

impl TypeWalk for Ty {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
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
            Ty::Dyn(predicates) | Ty::Opaque(predicates) => {
                for p in predicates.iter() {
                    p.walk(f);
                }
            }
            Ty::Param { .. } | Ty::Bound(_) | Ty::Infer(_) | Ty::Unknown => {}
        }
        f(self);
    }

    fn walk_mut_binders(&mut self, f: &mut impl FnMut(&mut Ty, usize), binders: usize) {
        match self {
            Ty::Apply(a_ty) => {
                a_ty.parameters.walk_mut_binders(f, binders);
            }
            Ty::Projection(p_ty) => {
                p_ty.parameters.walk_mut_binders(f, binders);
            }
            Ty::Dyn(predicates) | Ty::Opaque(predicates) => {
                for p in make_mut_slice(predicates) {
                    p.walk_mut_binders(f, binders + 1);
                }
            }
            Ty::Param { .. } | Ty::Bound(_) | Ty::Infer(_) | Ty::Unknown => {}
        }
        f(self, binders);
    }
}

const TYPE_HINT_TRUNCATION: &str = "…";

impl HirDisplay for &Ty {
    fn hir_fmt(&self, f: &mut HirFormatter<impl HirDatabase>) -> fmt::Result {
        HirDisplay::hir_fmt(*self, f)
    }
}

impl HirDisplay for ApplicationTy {
    fn hir_fmt(&self, f: &mut HirFormatter<impl HirDatabase>) -> fmt::Result {
        if f.should_truncate() {
            return write!(f, "{}", TYPE_HINT_TRUNCATION);
        }

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
                    CallableDef::FunctionId(ff) => f.db.function_data(ff).name.clone(),
                    CallableDef::StructId(s) => f.db.struct_data(s).name.clone(),
                    CallableDef::EnumVariantId(e) => {
                        let enum_data = f.db.enum_data(e.parent);
                        enum_data.variants[e.local_id].name.clone()
                    }
                };
                match def {
                    CallableDef::FunctionId(_) => write!(f, "fn {}", name)?,
                    CallableDef::StructId(_) | CallableDef::EnumVariantId(_) => {
                        write!(f, "{}", name)?
                    }
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
                    AdtId::StructId(it) => f.db.struct_data(it).name.clone(),
                    AdtId::UnionId(it) => f.db.union_data(it).name.clone(),
                    AdtId::EnumId(it) => f.db.enum_data(it).name.clone(),
                };
                write!(f, "{}", name)?;
                if self.parameters.len() > 0 {
                    write!(f, "<")?;

                    let mut non_default_parameters = Vec::with_capacity(self.parameters.len());
                    let parameters_to_write = if f.omit_verbose_types() {
                        match self
                            .ctor
                            .as_generic_def()
                            .map(|generic_def_id| f.db.generic_defaults(generic_def_id))
                            .filter(|defaults| !defaults.is_empty())
                        {
                            Option::None => self.parameters.0.as_ref(),
                            Option::Some(default_parameters) => {
                                for (i, parameter) in self.parameters.iter().enumerate() {
                                    match (parameter, default_parameters.get(i)) {
                                        (&Ty::Unknown, _) | (_, None) => {
                                            non_default_parameters.push(parameter.clone())
                                        }
                                        (_, Some(default_parameter))
                                            if parameter != default_parameter =>
                                        {
                                            non_default_parameters.push(parameter.clone())
                                        }
                                        _ => (),
                                    }
                                }
                                &non_default_parameters
                            }
                        }
                    } else {
                        self.parameters.0.as_ref()
                    };

                    f.write_joined(parameters_to_write, ", ")?;
                    write!(f, ">")?;
                }
            }
            TypeCtor::AssociatedType(type_alias) => {
                let trait_ = match type_alias.lookup(f.db).container {
                    AssocContainerId::TraitId(it) => it,
                    _ => panic!("not an associated type"),
                };
                let trait_name = f.db.trait_data(trait_).name.clone();
                let name = f.db.type_alias_data(type_alias).name.clone();
                write!(f, "{}::{}", trait_name, name)?;
                if self.parameters.len() > 0 {
                    write!(f, "<")?;
                    f.write_joined(&*self.parameters.0, ", ")?;
                    write!(f, ">")?;
                }
            }
            TypeCtor::Closure { .. } => {
                let sig = self.parameters[0]
                    .callable_sig(f.db)
                    .expect("first closure parameter should contain signature");
                let return_type_hint = sig.ret().display(f.db);
                if sig.params().is_empty() {
                    write!(f, "|| -> {}", return_type_hint)?;
                } else if f.omit_verbose_types() {
                    write!(f, "|{}| -> {}", TYPE_HINT_TRUNCATION, return_type_hint)?;
                } else {
                    write!(f, "|")?;
                    f.write_joined(sig.params(), ", ")?;
                    write!(f, "| -> {}", return_type_hint)?;
                };
            }
        }
        Ok(())
    }
}

impl HirDisplay for ProjectionTy {
    fn hir_fmt(&self, f: &mut HirFormatter<impl HirDatabase>) -> fmt::Result {
        if f.should_truncate() {
            return write!(f, "{}", TYPE_HINT_TRUNCATION);
        }

        let trait_name = f.db.trait_data(self.trait_(f.db)).name.clone();
        write!(f, "<{} as {}", self.parameters[0].display(f.db), trait_name,)?;
        if self.parameters.len() > 1 {
            write!(f, "<")?;
            f.write_joined(&self.parameters[1..], ", ")?;
            write!(f, ">")?;
        }
        write!(f, ">::{}", f.db.type_alias_data(self.associated_ty).name)?;
        Ok(())
    }
}

impl HirDisplay for Ty {
    fn hir_fmt(&self, f: &mut HirFormatter<impl HirDatabase>) -> fmt::Result {
        if f.should_truncate() {
            return write!(f, "{}", TYPE_HINT_TRUNCATION);
        }

        match self {
            Ty::Apply(a_ty) => a_ty.hir_fmt(f)?,
            Ty::Projection(p_ty) => p_ty.hir_fmt(f)?,
            Ty::Param { name, .. } => write!(f, "{}", name)?,
            Ty::Bound(idx) => write!(f, "?{}", idx)?,
            Ty::Dyn(predicates) | Ty::Opaque(predicates) => {
                match self {
                    Ty::Dyn(_) => write!(f, "dyn ")?,
                    Ty::Opaque(_) => write!(f, "impl ")?,
                    _ => unreachable!(),
                };
                // Note: This code is written to produce nice results (i.e.
                // corresponding to surface Rust) for types that can occur in
                // actual Rust. It will have weird results if the predicates
                // aren't as expected (i.e. self types = $0, projection
                // predicates for a certain trait come after the Implemented
                // predicate for that trait).
                let mut first = true;
                let mut angle_open = false;
                for p in predicates.iter() {
                    match p {
                        GenericPredicate::Implemented(trait_ref) => {
                            if angle_open {
                                write!(f, ">")?;
                            }
                            if !first {
                                write!(f, " + ")?;
                            }
                            // We assume that the self type is $0 (i.e. the
                            // existential) here, which is the only thing that's
                            // possible in actual Rust, and hence don't print it
                            write!(f, "{}", f.db.trait_data(trait_ref.trait_).name.clone())?;
                            if trait_ref.substs.len() > 1 {
                                write!(f, "<")?;
                                f.write_joined(&trait_ref.substs[1..], ", ")?;
                                // there might be assoc type bindings, so we leave the angle brackets open
                                angle_open = true;
                            }
                        }
                        GenericPredicate::Projection(projection_pred) => {
                            // in types in actual Rust, these will always come
                            // after the corresponding Implemented predicate
                            if angle_open {
                                write!(f, ", ")?;
                            } else {
                                write!(f, "<")?;
                                angle_open = true;
                            }
                            let name =
                                f.db.type_alias_data(projection_pred.projection_ty.associated_ty)
                                    .name
                                    .clone();
                            write!(f, "{} = ", name)?;
                            projection_pred.ty.hir_fmt(f)?;
                        }
                        GenericPredicate::Error => {
                            if angle_open {
                                // impl Trait<X, {error}>
                                write!(f, ", ")?;
                            } else if !first {
                                // impl Trait + {error}
                                write!(f, " + ")?;
                            }
                            p.hir_fmt(f)?;
                        }
                    }
                    first = false;
                }
                if angle_open {
                    write!(f, ">")?;
                }
            }
            Ty::Unknown => write!(f, "{{unknown}}")?,
            Ty::Infer(..) => write!(f, "_")?,
        }
        Ok(())
    }
}

impl TraitRef {
    fn hir_fmt_ext(&self, f: &mut HirFormatter<impl HirDatabase>, use_as: bool) -> fmt::Result {
        if f.should_truncate() {
            return write!(f, "{}", TYPE_HINT_TRUNCATION);
        }

        self.substs[0].hir_fmt(f)?;
        if use_as {
            write!(f, " as ")?;
        } else {
            write!(f, ": ")?;
        }
        write!(f, "{}", f.db.trait_data(self.trait_).name.clone())?;
        if self.substs.len() > 1 {
            write!(f, "<")?;
            f.write_joined(&self.substs[1..], ", ")?;
            write!(f, ">")?;
        }
        Ok(())
    }
}

impl HirDisplay for TraitRef {
    fn hir_fmt(&self, f: &mut HirFormatter<impl HirDatabase>) -> fmt::Result {
        self.hir_fmt_ext(f, false)
    }
}

impl HirDisplay for &GenericPredicate {
    fn hir_fmt(&self, f: &mut HirFormatter<impl HirDatabase>) -> fmt::Result {
        HirDisplay::hir_fmt(*self, f)
    }
}

impl HirDisplay for GenericPredicate {
    fn hir_fmt(&self, f: &mut HirFormatter<impl HirDatabase>) -> fmt::Result {
        if f.should_truncate() {
            return write!(f, "{}", TYPE_HINT_TRUNCATION);
        }

        match self {
            GenericPredicate::Implemented(trait_ref) => trait_ref.hir_fmt(f)?,
            GenericPredicate::Projection(projection_pred) => {
                write!(f, "<")?;
                projection_pred.projection_ty.trait_ref(f.db).hir_fmt_ext(f, true)?;
                write!(
                    f,
                    ">::{} = {}",
                    f.db.type_alias_data(projection_pred.projection_ty.associated_ty).name,
                    projection_pred.ty.display(f.db)
                )?;
            }
            GenericPredicate::Error => write!(f, "{{error}}")?,
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
