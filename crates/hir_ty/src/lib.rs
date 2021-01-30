//! The type system. We currently use this to infer types for completion, hover
//! information and various assists.
#[allow(unused)]
macro_rules! eprintln {
    ($($tt:tt)*) => { stdx::eprintln!($($tt)*) };
}

mod autoderef;
pub mod primitive;
pub mod traits;
pub mod method_resolution;
mod op;
mod lower;
pub(crate) mod infer;
pub(crate) mod utils;

pub mod display;
pub mod db;
pub mod diagnostics;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod test_db;

use std::{iter, mem, ops::Deref, sync::Arc};

use base_db::{salsa, CrateId};
use hir_def::{
    expr::ExprId,
    type_ref::{Mutability, Rawness},
    AdtId, AssocContainerId, DefWithBodyId, FunctionId, GenericDefId, HasModule, LifetimeParamId,
    Lookup, TraitId, TypeAliasId, TypeParamId,
};
use itertools::Itertools;

use crate::{
    db::HirDatabase,
    display::HirDisplay,
    primitive::{FloatTy, IntTy},
    utils::{generics, make_mut_slice, Generics},
};

pub use autoderef::autoderef;
pub use infer::{InferTy, InferenceResult};
pub use lower::{
    associated_type_shorthand_candidates, callable_item_sig, CallableDefId, ImplTraitLoweringMode,
    TyDefId, TyLoweringContext, ValueTyDefId,
};
pub use traits::{InEnvironment, Obligation, ProjectionPredicate, TraitEnvironment};

pub use chalk_ir::{BoundVar, DebruijnIndex};

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub enum Lifetime {
    Parameter(LifetimeParamId),
    Static,
}

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
    Int(IntTy),

    /// A primitive floating-point type. For example, `f64`.
    Float(FloatTy),

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
    FnDef(CallableDefId),

    /// A pointer to a function.  Written as `fn() -> i32`.
    ///
    /// For example the type of `bar` here:
    ///
    /// ```
    /// fn foo() -> i32 { 1 }
    /// let bar: fn() -> i32 = foo;
    /// ```
    // FIXME make this a Ty variant like in Chalk
    FnPtr { num_args: u16, is_varargs: bool },

    /// The never type `!`.
    Never,

    /// A tuple type.  For example, `(i32, bool)`.
    Tuple { cardinality: u16 },

    /// Represents an associated item like `Iterator::Item`.  This is used
    /// when we have tried to normalize a projection like `T::Item` but
    /// couldn't find a better representation.  In that case, we generate
    /// an **application type** like `(Iterator::Item)<T>`.
    AssociatedType(TypeAliasId),

    /// This represents a placeholder for an opaque type in situations where we
    /// don't know the hidden type (i.e. currently almost always). This is
    /// analogous to the `AssociatedType` type constructor.
    /// It is also used as the type of async block, with one type parameter
    /// representing the Future::Output type.
    OpaqueType(OpaqueTyId),

    /// Represents a foreign type declared in external blocks.
    ForeignType(TypeAliasId),

    /// The type of a specific closure.
    ///
    /// The closure signature is stored in a `FnPtr` type in the first type
    /// parameter.
    Closure { def: DefWithBodyId, expr: ExprId },
}

impl TypeCtor {
    pub fn num_ty_params(self, db: &dyn HirDatabase) -> usize {
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
                let generic_params = generics(db.upcast(), adt.into());
                generic_params.len()
            }
            TypeCtor::FnDef(callable) => {
                let generic_params = generics(db.upcast(), callable.into());
                generic_params.len()
            }
            TypeCtor::AssociatedType(type_alias) => {
                let generic_params = generics(db.upcast(), type_alias.into());
                generic_params.len()
            }
            TypeCtor::ForeignType(type_alias) => {
                let generic_params = generics(db.upcast(), type_alias.into());
                generic_params.len()
            }
            TypeCtor::OpaqueType(opaque_ty_id) => {
                match opaque_ty_id {
                    OpaqueTyId::ReturnTypeImplTrait(func, _) => {
                        let generic_params = generics(db.upcast(), func.into());
                        generic_params.len()
                    }
                    // 1 param representing Future::Output type.
                    OpaqueTyId::AsyncBlockTypeImplTrait(..) => 1,
                }
            }
            TypeCtor::FnPtr { num_args, is_varargs: _ } => num_args as usize + 1,
            TypeCtor::Tuple { cardinality } => cardinality as usize,
        }
    }

    pub fn krate(self, db: &dyn HirDatabase) -> Option<CrateId> {
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
            TypeCtor::Adt(adt) => Some(adt.module(db.upcast()).krate()),
            TypeCtor::FnDef(callable) => Some(callable.krate(db)),
            TypeCtor::AssociatedType(type_alias) => {
                Some(type_alias.lookup(db.upcast()).module(db.upcast()).krate())
            }
            TypeCtor::ForeignType(type_alias) => {
                Some(type_alias.lookup(db.upcast()).module(db.upcast()).krate())
            }
            TypeCtor::OpaqueType(opaque_ty_id) => match opaque_ty_id {
                OpaqueTyId::ReturnTypeImplTrait(func, _) => {
                    Some(func.lookup(db.upcast()).module(db.upcast()).krate())
                }
                OpaqueTyId::AsyncBlockTypeImplTrait(def, _) => {
                    Some(def.module(db.upcast()).krate())
                }
            },
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
            TypeCtor::ForeignType(type_alias) => Some(type_alias.into()),
            TypeCtor::OpaqueType(_impl_trait_id) => None,
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

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct OpaqueTy {
    pub opaque_ty_id: OpaqueTyId,
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
    pub fn trait_ref(&self, db: &dyn HirDatabase) -> TraitRef {
        TraitRef { trait_: self.trait_(db), substs: self.parameters.clone() }
    }

    fn trait_(&self, db: &dyn HirDatabase) -> TraitId {
        match self.associated_ty.lookup(db.upcast()).container {
            AssocContainerId::TraitId(it) => it,
            _ => panic!("projection ty without parent trait"),
        }
    }
}

impl TypeWalk for ProjectionTy {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        self.parameters.walk(f);
    }

    fn walk_mut_binders(
        &mut self,
        f: &mut impl FnMut(&mut Ty, DebruijnIndex),
        binders: DebruijnIndex,
    ) {
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

    /// An opaque type (`impl Trait`).
    ///
    /// This is currently only used for return type impl trait; each instance of
    /// `impl Trait` in a return type gets its own ID.
    Opaque(OpaqueTy),

    /// A placeholder for a type parameter; for example, `T` in `fn f<T>(x: T)
    /// {}` when we're type-checking the body of that function. In this
    /// situation, we know this stands for *some* type, but don't know the exact
    /// type.
    Placeholder(TypeParamId),

    /// A bound type variable. This is used in various places: when representing
    /// some polymorphic type like the type of function `fn f<T>`, the type
    /// parameters get turned into variables; during trait resolution, inference
    /// variables get turned into bound variables and back; and in `Dyn` the
    /// `Self` type is represented with a bound variable as well.
    Bound(BoundVar),

    /// A type variable used during type checking.
    Infer(InferTy),

    /// A trait object (`dyn Trait` or bare `Trait` in pre-2018 Rust).
    ///
    /// The predicates are quantified over the `Self` type, i.e. `Ty::Bound(0)`
    /// represents the `Self` type inside the bounds. This is currently
    /// implicit; Chalk has the `Binders` struct to make it explicit, but it
    /// didn't seem worth the overhead yet.
    Dyn(Arc<[GenericPredicate]>),

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

    fn walk_mut_binders(
        &mut self,
        f: &mut impl FnMut(&mut Ty, DebruijnIndex),
        binders: DebruijnIndex,
    ) {
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

    pub fn suffix(&self, n: usize) -> Substs {
        Substs(self.0[self.0.len() - std::cmp::min(self.0.len(), n)..].into())
    }

    pub fn as_single(&self) -> &Ty {
        if self.0.len() != 1 {
            panic!("expected substs of len 1, got {:?}", self);
        }
        &self.0[0]
    }

    /// Return Substs that replace each parameter by itself (i.e. `Ty::Param`).
    pub(crate) fn type_params_for_generics(generic_params: &Generics) -> Substs {
        Substs(generic_params.iter().map(|(id, _)| Ty::Placeholder(id)).collect())
    }

    /// Return Substs that replace each parameter by itself (i.e. `Ty::Param`).
    pub fn type_params(db: &dyn HirDatabase, def: impl Into<GenericDefId>) -> Substs {
        let params = generics(db.upcast(), def.into());
        Substs::type_params_for_generics(&params)
    }

    /// Return Substs that replace each parameter by a bound variable.
    pub(crate) fn bound_vars(generic_params: &Generics, debruijn: DebruijnIndex) -> Substs {
        Substs(
            generic_params
                .iter()
                .enumerate()
                .map(|(idx, _)| Ty::Bound(BoundVar::new(debruijn, idx)))
                .collect(),
        )
    }

    pub fn build_for_def(db: &dyn HirDatabase, def: impl Into<GenericDefId>) -> SubstsBuilder {
        let def = def.into();
        let params = generics(db.upcast(), def);
        let param_count = params.len();
        Substs::builder(param_count)
    }

    pub(crate) fn build_for_generics(generic_params: &Generics) -> SubstsBuilder {
        Substs::builder(generic_params.len())
    }

    pub fn build_for_type_ctor(db: &dyn HirDatabase, type_ctor: TypeCtor) -> SubstsBuilder {
        Substs::builder(type_ctor.num_ty_params(db))
    }

    fn builder(param_count: usize) -> SubstsBuilder {
        SubstsBuilder { vec: Vec::with_capacity(param_count), param_count }
    }
}

/// Return an index of a parameter in the generic type parameter list by it's id.
pub fn param_idx(db: &dyn HirDatabase, id: TypeParamId) -> Option<usize> {
    generics(db.upcast(), id.parent).param_idx(id)
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

    pub fn fill_with_bound_vars(self, debruijn: DebruijnIndex, starting_from: usize) -> Self {
        self.fill((starting_from..).map(|idx| Ty::Bound(BoundVar::new(debruijn, idx))))
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

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub struct Binders<T> {
    pub num_binders: usize,
    pub value: T,
}

impl<T> Binders<T> {
    pub fn new(num_binders: usize, value: T) -> Self {
        Self { num_binders, value }
    }

    pub fn as_ref(&self) -> Binders<&T> {
        Binders { num_binders: self.num_binders, value: &self.value }
    }

    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Binders<U> {
        Binders { num_binders: self.num_binders, value: f(self.value) }
    }

    pub fn filter_map<U>(self, f: impl FnOnce(T) -> Option<U>) -> Option<Binders<U>> {
        Some(Binders { num_binders: self.num_binders, value: f(self.value)? })
    }
}

impl<T: Clone> Binders<&T> {
    pub fn cloned(&self) -> Binders<T> {
        Binders { num_binders: self.num_binders, value: self.value.clone() }
    }
}

impl<T: TypeWalk> Binders<T> {
    /// Substitutes all variables.
    pub fn subst(self, subst: &Substs) -> T {
        assert_eq!(subst.len(), self.num_binders);
        self.value.subst_bound_vars(subst)
    }

    /// Substitutes just a prefix of the variables (shifting the rest).
    pub fn subst_prefix(self, subst: &Substs) -> Binders<T> {
        assert!(subst.len() < self.num_binders);
        Binders::new(self.num_binders - subst.len(), self.value.subst_bound_vars(subst))
    }
}

impl<T: TypeWalk> TypeWalk for Binders<T> {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        self.value.walk(f);
    }

    fn walk_mut_binders(
        &mut self,
        f: &mut impl FnMut(&mut Ty, DebruijnIndex),
        binders: DebruijnIndex,
    ) {
        self.value.walk_mut_binders(f, binders.shifted_in())
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

    fn walk_mut_binders(
        &mut self,
        f: &mut impl FnMut(&mut Ty, DebruijnIndex),
        binders: DebruijnIndex,
    ) {
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
        matches!(self, GenericPredicate::Error)
    }

    pub fn is_implemented(&self) -> bool {
        matches!(self, GenericPredicate::Implemented(_))
    }

    pub fn trait_ref(&self, db: &dyn HirDatabase) -> Option<TraitRef> {
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

    fn walk_mut_binders(
        &mut self,
        f: &mut impl FnMut(&mut Ty, DebruijnIndex),
        binders: DebruijnIndex,
    ) {
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
/// contained have been replaced by bound variables, and `kinds` tells us how
/// many there are and whether they were normal or float/int variables. This is
/// used to erase irrelevant differences between types before using them in
/// queries.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Canonical<T> {
    pub value: T,
    pub kinds: Arc<[TyKind]>,
}

impl<T> Canonical<T> {
    pub fn new(value: T, kinds: impl IntoIterator<Item = TyKind>) -> Self {
        Self { value, kinds: kinds.into_iter().collect() }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TyKind {
    General,
    Integer,
    Float,
}

/// A function signature as seen by type inference: Several parameter types and
/// one return type.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FnSig {
    params_and_return: Arc<[Ty]>,
    is_varargs: bool,
}

/// A polymorphic function signature.
pub type PolyFnSig = Binders<FnSig>;

impl FnSig {
    pub fn from_params_and_return(mut params: Vec<Ty>, ret: Ty, is_varargs: bool) -> FnSig {
        params.push(ret);
        FnSig { params_and_return: params.into(), is_varargs }
    }

    pub fn from_fn_ptr_substs(substs: &Substs, is_varargs: bool) -> FnSig {
        FnSig { params_and_return: Arc::clone(&substs.0), is_varargs }
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

    fn walk_mut_binders(
        &mut self,
        f: &mut impl FnMut(&mut Ty, DebruijnIndex),
        binders: DebruijnIndex,
    ) {
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
    pub fn fn_ptr(sig: FnSig) -> Self {
        Ty::apply(
            TypeCtor::FnPtr { num_args: sig.params().len() as u16, is_varargs: sig.is_varargs },
            Substs(sig.params_and_return),
        )
    }

    pub fn as_reference(&self) -> Option<(&Ty, Mutability)> {
        match self {
            Ty::Apply(ApplicationTy { ctor: TypeCtor::Ref(mutability), parameters }) => {
                Some((parameters.as_single(), *mutability))
            }
            _ => None,
        }
    }

    pub fn as_reference_or_ptr(&self) -> Option<(&Ty, Rawness, Mutability)> {
        match self {
            Ty::Apply(ApplicationTy { ctor: TypeCtor::Ref(mutability), parameters }) => {
                Some((parameters.as_single(), Rawness::Ref, *mutability))
            }
            Ty::Apply(ApplicationTy { ctor: TypeCtor::RawPtr(mutability), parameters }) => {
                Some((parameters.as_single(), Rawness::RawPtr, *mutability))
            }
            _ => None,
        }
    }

    pub fn strip_references(&self) -> &Ty {
        let mut t: &Ty = self;

        while let Ty::Apply(ApplicationTy { ctor: TypeCtor::Ref(_mutability), parameters }) = t {
            t = parameters.as_single();
        }

        t
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

    pub fn is_never(&self) -> bool {
        matches!(self, Ty::Apply(ApplicationTy { ctor: TypeCtor::Never, .. }))
    }

    pub fn is_unknown(&self) -> bool {
        matches!(self, Ty::Unknown)
    }

    /// If this is a `dyn Trait` type, this returns the `Trait` part.
    pub fn dyn_trait_ref(&self) -> Option<&TraitRef> {
        match self {
            Ty::Dyn(bounds) => bounds.get(0).and_then(|b| match b {
                GenericPredicate::Implemented(trait_ref) => Some(trait_ref),
                _ => None,
            }),
            _ => None,
        }
    }

    /// If this is a `dyn Trait`, returns that trait.
    pub fn dyn_trait(&self) -> Option<TraitId> {
        self.dyn_trait_ref().map(|it| it.trait_)
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

    pub fn as_fn_def(&self) -> Option<FunctionId> {
        match self {
            &Ty::Apply(ApplicationTy {
                ctor: TypeCtor::FnDef(CallableDefId::FunctionId(func)),
                ..
            }) => Some(func),
            _ => None,
        }
    }

    pub fn callable_sig(&self, db: &dyn HirDatabase) -> Option<FnSig> {
        match self {
            Ty::Apply(a_ty) => match a_ty.ctor {
                TypeCtor::FnPtr { is_varargs, .. } => {
                    Some(FnSig::from_fn_ptr_substs(&a_ty.parameters, is_varargs))
                }
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

    pub fn impl_trait_bounds(&self, db: &dyn HirDatabase) -> Option<Vec<GenericPredicate>> {
        match self {
            Ty::Apply(ApplicationTy { ctor: TypeCtor::OpaqueType(opaque_ty_id), .. }) => {
                match opaque_ty_id {
                    OpaqueTyId::AsyncBlockTypeImplTrait(def, _expr) => {
                        let krate = def.module(db.upcast()).krate();
                        if let Some(future_trait) = db
                            .lang_item(krate, "future_trait".into())
                            .and_then(|item| item.as_trait())
                        {
                            // This is only used by type walking.
                            // Parameters will be walked outside, and projection predicate is not used.
                            // So just provide the Future trait.
                            let impl_bound = GenericPredicate::Implemented(TraitRef {
                                trait_: future_trait,
                                substs: Substs::empty(),
                            });
                            Some(vec![impl_bound])
                        } else {
                            None
                        }
                    }
                    OpaqueTyId::ReturnTypeImplTrait(..) => None,
                }
            }
            Ty::Opaque(opaque_ty) => {
                let predicates = match opaque_ty.opaque_ty_id {
                    OpaqueTyId::ReturnTypeImplTrait(func, idx) => {
                        db.return_type_impl_traits(func).map(|it| {
                            let data = (*it)
                                .as_ref()
                                .map(|rpit| rpit.impl_traits[idx as usize].bounds.clone());
                            data.subst(&opaque_ty.parameters)
                        })
                    }
                    // It always has an parameter for Future::Output type.
                    OpaqueTyId::AsyncBlockTypeImplTrait(..) => unreachable!(),
                };

                predicates.map(|it| it.value)
            }
            Ty::Placeholder(id) => {
                let generic_params = db.generic_params(id.parent);
                let param_data = &generic_params.types[id.local_id];
                match param_data.provenance {
                    hir_def::generics::TypeParamProvenance::ArgumentImplTrait => {
                        let predicates = db
                            .generic_predicates_for_param(*id)
                            .into_iter()
                            .map(|pred| pred.value.clone())
                            .collect_vec();

                        Some(predicates)
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    pub fn associated_type_parent_trait(&self, db: &dyn HirDatabase) -> Option<TraitId> {
        match self {
            Ty::Apply(ApplicationTy { ctor: TypeCtor::AssociatedType(type_alias_id), .. }) => {
                match type_alias_id.lookup(db.upcast()).container {
                    AssocContainerId::TraitId(trait_id) => Some(trait_id),
                    _ => None,
                }
            }
            Ty::Projection(projection_ty) => {
                match projection_ty.associated_ty.lookup(db.upcast()).container {
                    AssocContainerId::TraitId(trait_id) => Some(trait_id),
                    _ => None,
                }
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
        self.walk_mut_binders(&mut |ty, _binders| f(ty), DebruijnIndex::INNERMOST);
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
    fn walk_mut_binders(
        &mut self,
        f: &mut impl FnMut(&mut Ty, DebruijnIndex),
        binders: DebruijnIndex,
    );

    fn fold_binders(
        mut self,
        f: &mut impl FnMut(Ty, DebruijnIndex) -> Ty,
        binders: DebruijnIndex,
    ) -> Self
    where
        Self: Sized,
    {
        self.walk_mut_binders(
            &mut |ty_mut, binders| {
                let ty = mem::replace(ty_mut, Ty::Unknown);
                *ty_mut = f(ty, binders);
            },
            binders,
        );
        self
    }

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

    /// Substitutes `Ty::Bound` vars with the given substitution.
    fn subst_bound_vars(self, substs: &Substs) -> Self
    where
        Self: Sized,
    {
        self.subst_bound_vars_at_depth(substs, DebruijnIndex::INNERMOST)
    }

    /// Substitutes `Ty::Bound` vars with the given substitution.
    fn subst_bound_vars_at_depth(mut self, substs: &Substs, depth: DebruijnIndex) -> Self
    where
        Self: Sized,
    {
        self.walk_mut_binders(
            &mut |ty, binders| {
                if let &mut Ty::Bound(bound) = ty {
                    if bound.debruijn >= binders {
                        *ty = substs.0[bound.index].clone().shift_bound_vars(binders);
                    }
                }
            },
            depth,
        );
        self
    }

    /// Shifts up debruijn indices of `Ty::Bound` vars by `n`.
    fn shift_bound_vars(self, n: DebruijnIndex) -> Self
    where
        Self: Sized,
    {
        self.fold_binders(
            &mut |ty, binders| match ty {
                Ty::Bound(bound) if bound.debruijn >= binders => {
                    Ty::Bound(bound.shifted_in_from(n))
                }
                ty => ty,
            },
            DebruijnIndex::INNERMOST,
        )
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
            Ty::Dyn(predicates) => {
                for p in predicates.iter() {
                    p.walk(f);
                }
            }
            Ty::Opaque(o_ty) => {
                for t in o_ty.parameters.iter() {
                    t.walk(f);
                }
            }
            Ty::Placeholder { .. } | Ty::Bound(_) | Ty::Infer(_) | Ty::Unknown => {}
        }
        f(self);
    }

    fn walk_mut_binders(
        &mut self,
        f: &mut impl FnMut(&mut Ty, DebruijnIndex),
        binders: DebruijnIndex,
    ) {
        match self {
            Ty::Apply(a_ty) => {
                a_ty.parameters.walk_mut_binders(f, binders);
            }
            Ty::Projection(p_ty) => {
                p_ty.parameters.walk_mut_binders(f, binders);
            }
            Ty::Dyn(predicates) => {
                for p in make_mut_slice(predicates) {
                    p.walk_mut_binders(f, binders.shifted_in());
                }
            }
            Ty::Opaque(o_ty) => {
                o_ty.parameters.walk_mut_binders(f, binders);
            }
            Ty::Placeholder { .. } | Ty::Bound(_) | Ty::Infer(_) | Ty::Unknown => {}
        }
        f(self, binders);
    }
}

impl<T: TypeWalk> TypeWalk for Vec<T> {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        for t in self {
            t.walk(f);
        }
    }
    fn walk_mut_binders(
        &mut self,
        f: &mut impl FnMut(&mut Ty, DebruijnIndex),
        binders: DebruijnIndex,
    ) {
        for t in self {
            t.walk_mut_binders(f, binders);
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub enum OpaqueTyId {
    ReturnTypeImplTrait(hir_def::FunctionId, u16),
    AsyncBlockTypeImplTrait(hir_def::DefWithBodyId, ExprId),
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct ReturnTypeImplTraits {
    pub(crate) impl_traits: Vec<ReturnTypeImplTrait>,
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub(crate) struct ReturnTypeImplTrait {
    pub(crate) bounds: Binders<Vec<GenericPredicate>>,
}
