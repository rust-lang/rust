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
mod chalk_cast;

pub mod display;
pub mod db;
pub mod diagnostics;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod test_db;
mod chalk_ext;

use std::{iter, mem, sync::Arc};

use base_db::salsa;
use chalk_ir::{
    cast::{CastTo, Caster},
    interner::HasInterner,
};
use hir_def::{
    builtin_type::BuiltinType, expr::ExprId, type_ref::Rawness, AssocContainerId, FunctionId,
    GenericDefId, HasModule, LifetimeParamId, Lookup, TraitId, TypeAliasId, TypeParamId,
};
use itertools::Itertools;
use smallvec::SmallVec;

use crate::{
    db::HirDatabase,
    display::HirDisplay,
    utils::{generics, make_mut_slice, Generics},
};

pub use autoderef::autoderef;
pub use chalk_ext::TyExt;
pub use infer::{could_unify, InferenceResult, InferenceVar};
pub use lower::{
    associated_type_shorthand_candidates, callable_item_sig, CallableDefId, ImplTraitLoweringMode,
    TyDefId, TyLoweringContext, ValueTyDefId,
};
pub use traits::{AliasEq, DomainGoal, InEnvironment, TraitEnvironment};

pub use chalk_ir::{
    cast::Cast, AdtId, BoundVar, DebruijnIndex, Mutability, Safety, Scalar, TyVariableKind,
};

pub use crate::traits::chalk::Interner;

pub type ForeignDefId = chalk_ir::ForeignDefId<Interner>;
pub type AssocTypeId = chalk_ir::AssocTypeId<Interner>;
pub type FnDefId = chalk_ir::FnDefId<Interner>;
pub type ClosureId = chalk_ir::ClosureId<Interner>;
pub type OpaqueTyId = chalk_ir::OpaqueTyId<Interner>;
pub type PlaceholderIndex = chalk_ir::PlaceholderIndex;

pub type CanonicalVarKinds = chalk_ir::CanonicalVarKinds<Interner>;

pub type ChalkTraitId = chalk_ir::TraitId<Interner>;

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub enum Lifetime {
    Parameter(LifetimeParamId),
    Static,
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct OpaqueTy {
    pub opaque_ty_id: OpaqueTyId,
    pub substitution: Substitution,
}

impl TypeWalk for OpaqueTy {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        self.substitution.walk(f);
    }

    fn walk_mut_binders(
        &mut self,
        f: &mut impl FnMut(&mut Ty, DebruijnIndex),
        binders: DebruijnIndex,
    ) {
        self.substitution.walk_mut_binders(f, binders);
    }
}

/// A "projection" type corresponds to an (unnormalized)
/// projection like `<P0 as Trait<P1..Pn>>::Foo`. Note that the
/// trait and all its parameters are fully known.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct ProjectionTy {
    pub associated_ty_id: AssocTypeId,
    pub substitution: Substitution,
}

impl ProjectionTy {
    pub fn trait_ref(&self, db: &dyn HirDatabase) -> TraitRef {
        TraitRef {
            trait_id: to_chalk_trait_id(self.trait_(db)),
            substitution: self.substitution.clone(),
        }
    }

    pub fn self_type_parameter(&self) -> &Ty {
        &self.substitution.interned(&Interner)[0].assert_ty_ref(&Interner)
    }

    fn trait_(&self, db: &dyn HirDatabase) -> TraitId {
        match from_assoc_type_id(self.associated_ty_id).lookup(db.upcast()).container {
            AssocContainerId::TraitId(it) => it,
            _ => panic!("projection ty without parent trait"),
        }
    }
}

impl TypeWalk for ProjectionTy {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        self.substitution.walk(f);
    }

    fn walk_mut_binders(
        &mut self,
        f: &mut impl FnMut(&mut Ty, DebruijnIndex),
        binders: DebruijnIndex,
    ) {
        self.substitution.walk_mut_binders(f, binders);
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct DynTy {
    /// The unknown self type.
    pub bounds: Binders<QuantifiedWhereClauses>,
}

pub type FnSig = chalk_ir::FnSig<Interner>;

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct FnPointer {
    pub num_args: usize,
    pub sig: FnSig,
    pub substs: Substitution,
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub enum AliasTy {
    /// A "projection" type corresponds to an (unnormalized)
    /// projection like `<P0 as Trait<P1..Pn>>::Foo`. Note that the
    /// trait and all its parameters are fully known.
    Projection(ProjectionTy),
    /// An opaque type (`impl Trait`).
    ///
    /// This is currently only used for return type impl trait; each instance of
    /// `impl Trait` in a return type gets its own ID.
    Opaque(OpaqueTy),
}

impl TypeWalk for AliasTy {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        match self {
            AliasTy::Projection(it) => it.walk(f),
            AliasTy::Opaque(it) => it.walk(f),
        }
    }

    fn walk_mut_binders(
        &mut self,
        f: &mut impl FnMut(&mut Ty, DebruijnIndex),
        binders: DebruijnIndex,
    ) {
        match self {
            AliasTy::Projection(it) => it.walk_mut_binders(f, binders),
            AliasTy::Opaque(it) => it.walk_mut_binders(f, binders),
        }
    }
}
/// A type.
///
/// See also the `TyKind` enum in rustc (librustc/ty/sty.rs), which represents
/// the same thing (but in a different way).
///
/// This should be cheap to clone.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub enum TyKind {
    /// Structures, enumerations and unions.
    Adt(AdtId<Interner>, Substitution),

    /// Represents an associated item like `Iterator::Item`.  This is used
    /// when we have tried to normalize a projection like `T::Item` but
    /// couldn't find a better representation.  In that case, we generate
    /// an **application type** like `(Iterator::Item)<T>`.
    AssociatedType(AssocTypeId, Substitution),

    /// a scalar type like `bool` or `u32`
    Scalar(Scalar),

    /// A tuple type.  For example, `(i32, bool)`.
    Tuple(usize, Substitution),

    /// An array with the given length. Written as `[T; n]`.
    Array(Ty),

    /// The pointee of an array slice.  Written as `[T]`.
    Slice(Ty),

    /// A raw pointer. Written as `*mut T` or `*const T`
    Raw(Mutability, Ty),

    /// A reference; a pointer with an associated lifetime. Written as
    /// `&'a mut T` or `&'a T`.
    Ref(Mutability, Ty),

    /// This represents a placeholder for an opaque type in situations where we
    /// don't know the hidden type (i.e. currently almost always). This is
    /// analogous to the `AssociatedType` type constructor.
    /// It is also used as the type of async block, with one type parameter
    /// representing the Future::Output type.
    OpaqueType(OpaqueTyId, Substitution),

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
    FnDef(FnDefId, Substitution),

    /// The pointee of a string slice. Written as `str`.
    Str,

    /// The never type `!`.
    Never,

    /// The type of a specific closure.
    ///
    /// The closure signature is stored in a `FnPtr` type in the first type
    /// parameter.
    Closure(ClosureId, Substitution),

    /// Represents a foreign type declared in external blocks.
    ForeignType(ForeignDefId),

    /// A pointer to a function.  Written as `fn() -> i32`.
    ///
    /// For example the type of `bar` here:
    ///
    /// ```
    /// fn foo() -> i32 { 1 }
    /// let bar: fn() -> i32 = foo;
    /// ```
    Function(FnPointer),

    /// An "alias" type represents some form of type alias, such as:
    /// - An associated type projection like `<T as Iterator>::Item`
    /// - `impl Trait` types
    /// - Named type aliases like `type Foo<X> = Vec<X>`
    Alias(AliasTy),

    /// A placeholder for a type parameter; for example, `T` in `fn f<T>(x: T)
    /// {}` when we're type-checking the body of that function. In this
    /// situation, we know this stands for *some* type, but don't know the exact
    /// type.
    Placeholder(PlaceholderIndex),

    /// A bound type variable. This is used in various places: when representing
    /// some polymorphic type like the type of function `fn f<T>`, the type
    /// parameters get turned into variables; during trait resolution, inference
    /// variables get turned into bound variables and back; and in `Dyn` the
    /// `Self` type is represented with a bound variable as well.
    BoundVar(BoundVar),

    /// A type variable used during type checking.
    InferenceVar(InferenceVar, TyVariableKind),

    /// A trait object (`dyn Trait` or bare `Trait` in pre-2018 Rust).
    ///
    /// The predicates are quantified over the `Self` type, i.e. `Ty::Bound(0)`
    /// represents the `Self` type inside the bounds. This is currently
    /// implicit; Chalk has the `Binders` struct to make it explicit, but it
    /// didn't seem worth the overhead yet.
    Dyn(DynTy),

    /// A placeholder for a type which could not be computed; this is propagated
    /// to avoid useless error messages. Doubles as a placeholder where type
    /// variables are inserted before type checking, since we want to try to
    /// infer a better type here anyway -- for the IDE use case, we want to try
    /// to infer as much as possible even in the presence of type errors.
    Unknown,
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct Ty(Arc<TyKind>);

impl TyKind {
    pub fn intern(self, _interner: &Interner) -> Ty {
        Ty(Arc::new(self))
    }
}

impl Ty {
    pub fn kind(&self, _interner: &Interner) -> &TyKind {
        &self.0
    }

    pub fn interned_mut(&mut self) -> &mut TyKind {
        Arc::make_mut(&mut self.0)
    }

    pub fn into_inner(self) -> TyKind {
        Arc::try_unwrap(self.0).unwrap_or_else(|a| (*a).clone())
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct GenericArg {
    interned: GenericArgData,
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub enum GenericArgData {
    Ty(Ty),
}

impl GenericArg {
    /// Constructs a generic argument using `GenericArgData`.
    pub fn new(_interner: &Interner, data: GenericArgData) -> Self {
        GenericArg { interned: data }
    }

    /// Gets the interned value.
    pub fn interned(&self) -> &GenericArgData {
        &self.interned
    }

    /// Asserts that this is a type argument.
    pub fn assert_ty_ref(&self, interner: &Interner) -> &Ty {
        self.ty(interner).unwrap()
    }

    /// Checks whether the generic argument is a type.
    pub fn is_ty(&self, _interner: &Interner) -> bool {
        match self.interned() {
            GenericArgData::Ty(_) => true,
        }
    }

    /// Returns the type if it is one, `None` otherwise.
    pub fn ty(&self, _interner: &Interner) -> Option<&Ty> {
        match self.interned() {
            GenericArgData::Ty(t) => Some(t),
        }
    }
}

impl TypeWalk for GenericArg {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        match &self.interned {
            GenericArgData::Ty(ty) => {
                ty.walk(f);
            }
        }
    }

    fn walk_mut_binders(
        &mut self,
        f: &mut impl FnMut(&mut Ty, DebruijnIndex),
        binders: DebruijnIndex,
    ) {
        match &mut self.interned {
            GenericArgData::Ty(ty) => {
                ty.walk_mut_binders(f, binders);
            }
        }
    }
}

/// A list of substitutions for generic parameters.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct Substitution(SmallVec<[GenericArg; 2]>);

impl TypeWalk for Substitution {
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
        for t in &mut self.0 {
            t.walk_mut_binders(f, binders);
        }
    }
}

impl Substitution {
    pub fn interned(&self, _: &Interner) -> &[GenericArg] {
        &self.0
    }

    pub fn len(&self, _: &Interner) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self, _: &Interner) -> bool {
        self.0.is_empty()
    }

    pub fn at(&self, _: &Interner, i: usize) -> &GenericArg {
        &self.0[i]
    }

    pub fn empty(_: &Interner) -> Substitution {
        Substitution(SmallVec::new())
    }

    pub fn iter(&self, _: &Interner) -> std::slice::Iter<'_, GenericArg> {
        self.0.iter()
    }

    pub fn single(ty: Ty) -> Substitution {
        Substitution({
            let mut v = SmallVec::new();
            v.push(ty.cast(&Interner));
            v
        })
    }

    pub fn prefix(&self, n: usize) -> Substitution {
        Substitution(self.0[..std::cmp::min(self.0.len(), n)].into())
    }

    pub fn suffix(&self, n: usize) -> Substitution {
        Substitution(self.0[self.0.len() - std::cmp::min(self.0.len(), n)..].into())
    }

    pub fn from_iter(
        interner: &Interner,
        elements: impl IntoIterator<Item = impl CastTo<GenericArg>>,
    ) -> Self {
        Substitution(elements.into_iter().casted(interner).collect())
    }

    /// Return Substs that replace each parameter by itself (i.e. `Ty::Param`).
    pub(crate) fn type_params_for_generics(
        db: &dyn HirDatabase,
        generic_params: &Generics,
    ) -> Substitution {
        Substitution::from_iter(
            &Interner,
            generic_params
                .iter()
                .map(|(id, _)| TyKind::Placeholder(to_placeholder_idx(db, id)).intern(&Interner)),
        )
    }

    /// Return Substs that replace each parameter by itself (i.e. `Ty::Param`).
    pub fn type_params(db: &dyn HirDatabase, def: impl Into<GenericDefId>) -> Substitution {
        let params = generics(db.upcast(), def.into());
        Substitution::type_params_for_generics(db, &params)
    }

    /// Return Substs that replace each parameter by a bound variable.
    pub(crate) fn bound_vars(generic_params: &Generics, debruijn: DebruijnIndex) -> Substitution {
        Substitution::from_iter(
            &Interner,
            generic_params
                .iter()
                .enumerate()
                .map(|(idx, _)| TyKind::BoundVar(BoundVar::new(debruijn, idx)).intern(&Interner)),
        )
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
    vec: Vec<GenericArg>,
    param_count: usize,
}

impl SubstsBuilder {
    pub fn build(self) -> Substitution {
        assert_eq!(self.vec.len(), self.param_count);
        Substitution::from_iter(&Interner, self.vec)
    }

    pub fn push(mut self, ty: impl CastTo<GenericArg>) -> Self {
        self.vec.push(ty.cast(&Interner));
        self
    }

    fn remaining(&self) -> usize {
        self.param_count - self.vec.len()
    }

    pub fn fill_with_bound_vars(self, debruijn: DebruijnIndex, starting_from: usize) -> Self {
        self.fill(
            (starting_from..)
                .map(|idx| TyKind::BoundVar(BoundVar::new(debruijn, idx)).intern(&Interner)),
        )
    }

    pub fn fill_with_unknown(self) -> Self {
        self.fill(iter::repeat(TyKind::Unknown.intern(&Interner)))
    }

    pub fn fill(mut self, filler: impl Iterator<Item = impl CastTo<GenericArg>>) -> Self {
        self.vec.extend(filler.take(self.remaining()).casted(&Interner));
        assert_eq!(self.remaining(), 0);
        self
    }

    pub fn use_parent_substs(mut self, parent_substs: &Substitution) -> Self {
        assert!(self.vec.is_empty());
        assert!(parent_substs.len(&Interner) <= self.param_count);
        self.vec.extend(parent_substs.iter(&Interner).cloned());
        self
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

    pub fn wrap_empty(value: T) -> Self
    where
        T: TypeWalk,
    {
        Self { num_binders: 0, value: value.shift_bound_vars(DebruijnIndex::ONE) }
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

    pub fn skip_binders(&self) -> &T {
        &self.value
    }

    pub fn into_value_and_skipped_binders(self) -> (T, usize) {
        (self.value, self.num_binders)
    }
}

impl<T: Clone> Binders<&T> {
    pub fn cloned(&self) -> Binders<T> {
        Binders { num_binders: self.num_binders, value: self.value.clone() }
    }
}

impl<T: TypeWalk> Binders<T> {
    /// Substitutes all variables.
    pub fn subst(self, subst: &Substitution) -> T {
        assert_eq!(subst.len(&Interner), self.num_binders);
        self.value.subst_bound_vars(subst)
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
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct TraitRef {
    pub trait_id: ChalkTraitId,
    pub substitution: Substitution,
}

impl TraitRef {
    pub fn self_type_parameter(&self) -> &Ty {
        &self.substitution.at(&Interner, 0).assert_ty_ref(&Interner)
    }

    pub fn hir_trait_id(&self) -> TraitId {
        from_chalk_trait_id(self.trait_id)
    }
}

impl TypeWalk for TraitRef {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        self.substitution.walk(f);
    }

    fn walk_mut_binders(
        &mut self,
        f: &mut impl FnMut(&mut Ty, DebruijnIndex),
        binders: DebruijnIndex,
    ) {
        self.substitution.walk_mut_binders(f, binders);
    }
}

/// Like `generics::WherePredicate`, but with resolved types: A condition on the
/// parameters of a generic item.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum WhereClause {
    /// The given trait needs to be implemented for its type parameters.
    Implemented(TraitRef),
    /// An associated type bindings like in `Iterator<Item = T>`.
    AliasEq(AliasEq),
}

impl WhereClause {
    pub fn is_implemented(&self) -> bool {
        matches!(self, WhereClause::Implemented(_))
    }

    pub fn trait_ref(&self, db: &dyn HirDatabase) -> Option<TraitRef> {
        match self {
            WhereClause::Implemented(tr) => Some(tr.clone()),
            WhereClause::AliasEq(AliasEq { alias: AliasTy::Projection(proj), .. }) => {
                Some(proj.trait_ref(db))
            }
            WhereClause::AliasEq(_) => None,
        }
    }
}

impl TypeWalk for WhereClause {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        match self {
            WhereClause::Implemented(trait_ref) => trait_ref.walk(f),
            WhereClause::AliasEq(alias_eq) => alias_eq.walk(f),
        }
    }

    fn walk_mut_binders(
        &mut self,
        f: &mut impl FnMut(&mut Ty, DebruijnIndex),
        binders: DebruijnIndex,
    ) {
        match self {
            WhereClause::Implemented(trait_ref) => trait_ref.walk_mut_binders(f, binders),
            WhereClause::AliasEq(alias_eq) => alias_eq.walk_mut_binders(f, binders),
        }
    }
}

pub type QuantifiedWhereClause = Binders<WhereClause>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QuantifiedWhereClauses(Arc<[QuantifiedWhereClause]>);

impl QuantifiedWhereClauses {
    pub fn from_iter(
        _interner: &Interner,
        elements: impl IntoIterator<Item = QuantifiedWhereClause>,
    ) -> Self {
        QuantifiedWhereClauses(elements.into_iter().collect())
    }

    pub fn interned(&self) -> &Arc<[QuantifiedWhereClause]> {
        &self.0
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
    pub binders: CanonicalVarKinds,
}

impl<T> Canonical<T> {
    pub fn new(value: T, kinds: impl IntoIterator<Item = TyVariableKind>) -> Self {
        let kinds = kinds.into_iter().map(|tk| {
            chalk_ir::CanonicalVarKind::new(
                chalk_ir::VariableKind::Ty(tk),
                chalk_ir::UniverseIndex::ROOT,
            )
        });
        Self { value, binders: chalk_ir::CanonicalVarKinds::from_iter(&Interner, kinds) }
    }
}

/// A function signature as seen by type inference: Several parameter types and
/// one return type.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CallableSig {
    params_and_return: Arc<[Ty]>,
    is_varargs: bool,
}

/// A polymorphic function signature.
pub type PolyFnSig = Binders<CallableSig>;

impl CallableSig {
    pub fn from_params_and_return(mut params: Vec<Ty>, ret: Ty, is_varargs: bool) -> CallableSig {
        params.push(ret);
        CallableSig { params_and_return: params.into(), is_varargs }
    }

    pub fn from_fn_ptr(fn_ptr: &FnPointer) -> CallableSig {
        CallableSig {
            // FIXME: what to do about lifetime params? -> return PolyFnSig
            params_and_return: fn_ptr
                .substs
                .clone()
                .shift_bound_vars_out(DebruijnIndex::ONE)
                .interned(&Interner)
                .iter()
                .map(|arg| arg.assert_ty_ref(&Interner).clone())
                .collect(),
            is_varargs: fn_ptr.sig.variadic,
        }
    }

    pub fn params(&self) -> &[Ty] {
        &self.params_and_return[0..self.params_and_return.len() - 1]
    }

    pub fn ret(&self) -> &Ty {
        &self.params_and_return[self.params_and_return.len() - 1]
    }
}

impl TypeWalk for CallableSig {
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

pub struct TyBuilder<D> {
    data: D,
    vec: SmallVec<[GenericArg; 2]>,
    param_count: usize,
}

impl<D> TyBuilder<D> {
    fn new(data: D, param_count: usize) -> TyBuilder<D> {
        TyBuilder { data, param_count, vec: SmallVec::with_capacity(param_count) }
    }

    fn build_internal(self) -> (D, Substitution) {
        assert_eq!(self.vec.len(), self.param_count);
        // FIXME: would be good to have a way to construct a chalk_ir::Substitution from the interned form
        let subst = Substitution(self.vec);
        (self.data, subst)
    }

    pub fn push(mut self, arg: impl CastTo<GenericArg>) -> Self {
        self.vec.push(arg.cast(&Interner));
        self
    }

    fn remaining(&self) -> usize {
        self.param_count - self.vec.len()
    }

    pub fn fill_with_bound_vars(self, debruijn: DebruijnIndex, starting_from: usize) -> Self {
        self.fill(
            (starting_from..)
                .map(|idx| TyKind::BoundVar(BoundVar::new(debruijn, idx)).intern(&Interner)),
        )
    }

    pub fn fill_with_unknown(self) -> Self {
        self.fill(iter::repeat(TyKind::Unknown.intern(&Interner)))
    }

    pub fn fill(mut self, filler: impl Iterator<Item = impl CastTo<GenericArg>>) -> Self {
        self.vec.extend(filler.take(self.remaining()).casted(&Interner));
        assert_eq!(self.remaining(), 0);
        self
    }

    pub fn use_parent_substs(mut self, parent_substs: &Substitution) -> Self {
        assert!(self.vec.is_empty());
        assert!(parent_substs.len(&Interner) <= self.param_count);
        self.vec.extend(parent_substs.iter(&Interner).cloned());
        self
    }
}

impl TyBuilder<()> {
    pub fn unit() -> Ty {
        TyKind::Tuple(0, Substitution::empty(&Interner)).intern(&Interner)
    }

    pub fn fn_ptr(sig: CallableSig) -> Ty {
        TyKind::Function(FnPointer {
            num_args: sig.params().len(),
            sig: FnSig { abi: (), safety: Safety::Safe, variadic: sig.is_varargs },
            substs: Substitution::from_iter(&Interner, sig.params_and_return.iter().cloned()),
        })
        .intern(&Interner)
    }

    pub fn builtin(builtin: BuiltinType) -> Ty {
        match builtin {
            BuiltinType::Char => TyKind::Scalar(Scalar::Char).intern(&Interner),
            BuiltinType::Bool => TyKind::Scalar(Scalar::Bool).intern(&Interner),
            BuiltinType::Str => TyKind::Str.intern(&Interner),
            BuiltinType::Int(t) => {
                TyKind::Scalar(Scalar::Int(primitive::int_ty_from_builtin(t))).intern(&Interner)
            }
            BuiltinType::Uint(t) => {
                TyKind::Scalar(Scalar::Uint(primitive::uint_ty_from_builtin(t))).intern(&Interner)
            }
            BuiltinType::Float(t) => {
                TyKind::Scalar(Scalar::Float(primitive::float_ty_from_builtin(t))).intern(&Interner)
            }
        }
    }

    pub fn subst_for_def(db: &dyn HirDatabase, def: impl Into<GenericDefId>) -> TyBuilder<()> {
        let def = def.into();
        let params = generics(db.upcast(), def);
        let param_count = params.len();
        TyBuilder::new((), param_count)
    }

    pub fn build(self) -> Substitution {
        let ((), subst) = self.build_internal();
        subst
    }
}

impl TyBuilder<hir_def::AdtId> {
    pub fn adt(db: &dyn HirDatabase, adt: hir_def::AdtId) -> TyBuilder<hir_def::AdtId> {
        let generics = generics(db.upcast(), adt.into());
        let param_count = generics.len();
        TyBuilder::new(adt, param_count)
    }

    pub fn fill_with_defaults(
        mut self,
        db: &dyn HirDatabase,
        mut fallback: impl FnMut() -> Ty,
    ) -> Self {
        let defaults = db.generic_defaults(self.data.into());
        for default_ty in defaults.iter().skip(self.vec.len()) {
            if default_ty.skip_binders().is_unknown() {
                self.vec.push(fallback().cast(&Interner));
            } else {
                // each default can depend on the previous parameters
                let subst_so_far = Substitution(self.vec.clone());
                self.vec.push(default_ty.clone().subst(&subst_so_far).cast(&Interner));
            }
        }
        self
    }

    pub fn build(self) -> Ty {
        let (adt, subst) = self.build_internal();
        TyKind::Adt(AdtId(adt), subst).intern(&Interner)
    }
}

impl TyBuilder<TraitId> {
    pub fn trait_ref(db: &dyn HirDatabase, trait_id: TraitId) -> TyBuilder<TraitId> {
        let generics = generics(db.upcast(), trait_id.into());
        let param_count = generics.len();
        TyBuilder::new(trait_id, param_count)
    }

    pub fn build(self) -> TraitRef {
        let (trait_id, substitution) = self.build_internal();
        TraitRef { trait_id: to_chalk_trait_id(trait_id), substitution }
    }
}

impl TyBuilder<TypeAliasId> {
    pub fn assoc_type_projection(
        db: &dyn HirDatabase,
        type_alias: TypeAliasId,
    ) -> TyBuilder<TypeAliasId> {
        let generics = generics(db.upcast(), type_alias.into());
        let param_count = generics.len();
        TyBuilder::new(type_alias, param_count)
    }

    pub fn build(self) -> ProjectionTy {
        let (type_alias, substitution) = self.build_internal();
        ProjectionTy { associated_ty_id: to_assoc_type_id(type_alias), substitution }
    }
}

impl<T: TypeWalk + HasInterner<Interner = Interner>> TyBuilder<Binders<T>> {
    fn subst_binders(b: Binders<T>) -> Self {
        let param_count = b.num_binders;
        TyBuilder::new(b, param_count)
    }

    pub fn build(self) -> T {
        let (b, subst) = self.build_internal();
        b.subst(&subst)
    }
}

impl TyBuilder<Binders<Ty>> {
    pub fn def_ty(db: &dyn HirDatabase, def: TyDefId) -> TyBuilder<Binders<Ty>> {
        TyBuilder::subst_binders(db.ty(def.into()))
    }

    pub fn impl_self_ty(db: &dyn HirDatabase, def: hir_def::ImplId) -> TyBuilder<Binders<Ty>> {
        TyBuilder::subst_binders(db.impl_self_ty(def))
    }
}

impl Ty {
    pub fn as_reference(&self) -> Option<(&Ty, Mutability)> {
        match self.kind(&Interner) {
            TyKind::Ref(mutability, ty) => Some((ty, *mutability)),
            _ => None,
        }
    }

    pub fn as_reference_or_ptr(&self) -> Option<(&Ty, Rawness, Mutability)> {
        match self.kind(&Interner) {
            TyKind::Ref(mutability, ty) => Some((ty, Rawness::Ref, *mutability)),
            TyKind::Raw(mutability, ty) => Some((ty, Rawness::RawPtr, *mutability)),
            _ => None,
        }
    }

    pub fn strip_references(&self) -> &Ty {
        let mut t: &Ty = self;

        while let TyKind::Ref(_mutability, ty) = t.kind(&Interner) {
            t = ty;
        }

        t
    }

    pub fn as_adt(&self) -> Option<(hir_def::AdtId, &Substitution)> {
        match self.kind(&Interner) {
            TyKind::Adt(AdtId(adt), parameters) => Some((*adt, parameters)),
            _ => None,
        }
    }

    pub fn as_tuple(&self) -> Option<&Substitution> {
        match self.kind(&Interner) {
            TyKind::Tuple(_, substs) => Some(substs),
            _ => None,
        }
    }

    pub fn as_generic_def(&self, db: &dyn HirDatabase) -> Option<GenericDefId> {
        match *self.kind(&Interner) {
            TyKind::Adt(AdtId(adt), ..) => Some(adt.into()),
            TyKind::FnDef(callable, ..) => {
                Some(db.lookup_intern_callable_def(callable.into()).into())
            }
            TyKind::AssociatedType(type_alias, ..) => Some(from_assoc_type_id(type_alias).into()),
            TyKind::ForeignType(type_alias, ..) => Some(from_foreign_def_id(type_alias).into()),
            _ => None,
        }
    }

    pub fn is_never(&self) -> bool {
        matches!(self.kind(&Interner), TyKind::Never)
    }

    pub fn is_unknown(&self) -> bool {
        matches!(self.kind(&Interner), TyKind::Unknown)
    }

    pub fn equals_ctor(&self, other: &Ty) -> bool {
        match (self.kind(&Interner), other.kind(&Interner)) {
            (TyKind::Adt(adt, ..), TyKind::Adt(adt2, ..)) => adt == adt2,
            (TyKind::Slice(_), TyKind::Slice(_)) | (TyKind::Array(_), TyKind::Array(_)) => true,
            (TyKind::FnDef(def_id, ..), TyKind::FnDef(def_id2, ..)) => def_id == def_id2,
            (TyKind::OpaqueType(ty_id, ..), TyKind::OpaqueType(ty_id2, ..)) => ty_id == ty_id2,
            (TyKind::AssociatedType(ty_id, ..), TyKind::AssociatedType(ty_id2, ..)) => {
                ty_id == ty_id2
            }
            (TyKind::ForeignType(ty_id, ..), TyKind::ForeignType(ty_id2, ..)) => ty_id == ty_id2,
            (TyKind::Closure(id1, _), TyKind::Closure(id2, _)) => id1 == id2,
            (TyKind::Ref(mutability, ..), TyKind::Ref(mutability2, ..))
            | (TyKind::Raw(mutability, ..), TyKind::Raw(mutability2, ..)) => {
                mutability == mutability2
            }
            (
                TyKind::Function(FnPointer { num_args, sig, .. }),
                TyKind::Function(FnPointer { num_args: num_args2, sig: sig2, .. }),
            ) => num_args == num_args2 && sig == sig2,
            (TyKind::Tuple(cardinality, _), TyKind::Tuple(cardinality2, _)) => {
                cardinality == cardinality2
            }
            (TyKind::Str, TyKind::Str) | (TyKind::Never, TyKind::Never) => true,
            (TyKind::Scalar(scalar), TyKind::Scalar(scalar2)) => scalar == scalar2,
            _ => false,
        }
    }

    /// If this is a `dyn Trait` type, this returns the `Trait` part.
    fn dyn_trait_ref(&self) -> Option<&TraitRef> {
        match self.kind(&Interner) {
            TyKind::Dyn(dyn_ty) => {
                dyn_ty.bounds.value.interned().get(0).and_then(|b| match b.skip_binders() {
                    WhereClause::Implemented(trait_ref) => Some(trait_ref),
                    _ => None,
                })
            }
            _ => None,
        }
    }

    /// If this is a `dyn Trait`, returns that trait.
    pub fn dyn_trait(&self) -> Option<TraitId> {
        self.dyn_trait_ref().map(|it| it.trait_id).map(from_chalk_trait_id)
    }

    fn builtin_deref(&self) -> Option<Ty> {
        match self.kind(&Interner) {
            TyKind::Ref(.., ty) => Some(ty.clone()),
            TyKind::Raw(.., ty) => Some(ty.clone()),
            _ => None,
        }
    }

    pub fn callable_def(&self, db: &dyn HirDatabase) -> Option<CallableDefId> {
        match self.kind(&Interner) {
            &TyKind::FnDef(def, ..) => Some(db.lookup_intern_callable_def(def.into())),
            _ => None,
        }
    }

    pub fn as_fn_def(&self, db: &dyn HirDatabase) -> Option<FunctionId> {
        if let Some(CallableDefId::FunctionId(func)) = self.callable_def(db) {
            Some(func)
        } else {
            None
        }
    }

    pub fn callable_sig(&self, db: &dyn HirDatabase) -> Option<CallableSig> {
        match self.kind(&Interner) {
            TyKind::Function(fn_ptr) => Some(CallableSig::from_fn_ptr(fn_ptr)),
            TyKind::FnDef(def, parameters) => {
                let callable_def = db.lookup_intern_callable_def((*def).into());
                let sig = db.callable_item_signature(callable_def);
                Some(sig.subst(&parameters))
            }
            TyKind::Closure(.., substs) => {
                let sig_param = substs.at(&Interner, 0).assert_ty_ref(&Interner);
                sig_param.callable_sig(db)
            }
            _ => None,
        }
    }

    /// Returns the type parameters of this type if it has some (i.e. is an ADT
    /// or function); so if `self` is `Option<u32>`, this returns the `u32`.
    pub fn substs(&self) -> Option<&Substitution> {
        match self.kind(&Interner) {
            TyKind::Adt(_, substs)
            | TyKind::FnDef(_, substs)
            | TyKind::Function(FnPointer { substs, .. })
            | TyKind::Tuple(_, substs)
            | TyKind::OpaqueType(_, substs)
            | TyKind::AssociatedType(_, substs)
            | TyKind::Closure(.., substs) => Some(substs),
            _ => None,
        }
    }

    fn substs_mut(&mut self) -> Option<&mut Substitution> {
        match self.interned_mut() {
            TyKind::Adt(_, substs)
            | TyKind::FnDef(_, substs)
            | TyKind::Function(FnPointer { substs, .. })
            | TyKind::Tuple(_, substs)
            | TyKind::OpaqueType(_, substs)
            | TyKind::AssociatedType(_, substs)
            | TyKind::Closure(.., substs) => Some(substs),
            _ => None,
        }
    }

    pub fn impl_trait_bounds(&self, db: &dyn HirDatabase) -> Option<Vec<QuantifiedWhereClause>> {
        match self.kind(&Interner) {
            TyKind::OpaqueType(opaque_ty_id, ..) => {
                match db.lookup_intern_impl_trait_id((*opaque_ty_id).into()) {
                    ImplTraitId::AsyncBlockTypeImplTrait(def, _expr) => {
                        let krate = def.module(db.upcast()).krate();
                        if let Some(future_trait) = db
                            .lang_item(krate, "future_trait".into())
                            .and_then(|item| item.as_trait())
                        {
                            // This is only used by type walking.
                            // Parameters will be walked outside, and projection predicate is not used.
                            // So just provide the Future trait.
                            let impl_bound = Binders::new(
                                0,
                                WhereClause::Implemented(TraitRef {
                                    trait_id: to_chalk_trait_id(future_trait),
                                    substitution: Substitution::empty(&Interner),
                                }),
                            );
                            Some(vec![impl_bound])
                        } else {
                            None
                        }
                    }
                    ImplTraitId::ReturnTypeImplTrait(..) => None,
                }
            }
            TyKind::Alias(AliasTy::Opaque(opaque_ty)) => {
                let predicates = match db.lookup_intern_impl_trait_id(opaque_ty.opaque_ty_id.into())
                {
                    ImplTraitId::ReturnTypeImplTrait(func, idx) => {
                        db.return_type_impl_traits(func).map(|it| {
                            let data = (*it)
                                .as_ref()
                                .map(|rpit| rpit.impl_traits[idx as usize].bounds.clone());
                            data.subst(&opaque_ty.substitution)
                        })
                    }
                    // It always has an parameter for Future::Output type.
                    ImplTraitId::AsyncBlockTypeImplTrait(..) => unreachable!(),
                };

                predicates.map(|it| it.value)
            }
            TyKind::Placeholder(idx) => {
                let id = from_placeholder_idx(db, *idx);
                let generic_params = db.generic_params(id.parent);
                let param_data = &generic_params.types[id.local_id];
                match param_data.provenance {
                    hir_def::generics::TypeParamProvenance::ArgumentImplTrait => {
                        let substs = Substitution::type_params(db, id.parent);
                        let predicates = db
                            .generic_predicates(id.parent)
                            .into_iter()
                            .map(|pred| pred.clone().subst(&substs))
                            .filter(|wc| match &wc.skip_binders() {
                                WhereClause::Implemented(tr) => tr.self_type_parameter() == self,
                                WhereClause::AliasEq(AliasEq {
                                    alias: AliasTy::Projection(proj),
                                    ty: _,
                                }) => proj.self_type_parameter() == self,
                                _ => false,
                            })
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
        match self.kind(&Interner) {
            TyKind::AssociatedType(id, ..) => {
                match from_assoc_type_id(*id).lookup(db.upcast()).container {
                    AssocContainerId::TraitId(trait_id) => Some(trait_id),
                    _ => None,
                }
            }
            TyKind::Alias(AliasTy::Projection(projection_ty)) => {
                match from_assoc_type_id(projection_ty.associated_ty_id)
                    .lookup(db.upcast())
                    .container
                {
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
    /// `TyKind::Bound` variables use DeBruijn indexing, which means that 0 refers
    /// to the innermost binder, 1 to the next, etc.. So when we want to
    /// substitute a certain bound variable, we can't just walk the whole type
    /// and blindly replace each instance of a certain index; when we 'enter'
    /// things that introduce new bound variables, we have to keep track of
    /// that. Currently, the only thing that introduces bound variables on our
    /// side are `TyKind::Dyn` and `TyKind::Opaque`, which each introduce a bound
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
                let ty = mem::replace(ty_mut, TyKind::Unknown.intern(&Interner));
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
            let ty = mem::replace(ty_mut, TyKind::Unknown.intern(&Interner));
            *ty_mut = f(ty);
        });
        self
    }

    /// Substitutes `TyKind::Bound` vars with the given substitution.
    fn subst_bound_vars(self, substs: &Substitution) -> Self
    where
        Self: Sized,
    {
        self.subst_bound_vars_at_depth(substs, DebruijnIndex::INNERMOST)
    }

    /// Substitutes `TyKind::Bound` vars with the given substitution.
    fn subst_bound_vars_at_depth(mut self, substs: &Substitution, depth: DebruijnIndex) -> Self
    where
        Self: Sized,
    {
        self.walk_mut_binders(
            &mut |ty, binders| {
                if let &mut TyKind::BoundVar(bound) = ty.interned_mut() {
                    if bound.debruijn >= binders {
                        *ty = substs.0[bound.index]
                            .assert_ty_ref(&Interner)
                            .clone()
                            .shift_bound_vars(binders);
                    }
                }
            },
            depth,
        );
        self
    }

    /// Shifts up debruijn indices of `TyKind::Bound` vars by `n`.
    fn shift_bound_vars(self, n: DebruijnIndex) -> Self
    where
        Self: Sized,
    {
        self.fold_binders(
            &mut |ty, binders| match ty.kind(&Interner) {
                TyKind::BoundVar(bound) if bound.debruijn >= binders => {
                    TyKind::BoundVar(bound.shifted_in_from(n)).intern(&Interner)
                }
                _ => ty,
            },
            DebruijnIndex::INNERMOST,
        )
    }

    /// Shifts debruijn indices of `TyKind::Bound` vars out (down) by `n`.
    fn shift_bound_vars_out(self, n: DebruijnIndex) -> Self
    where
        Self: Sized + std::fmt::Debug,
    {
        self.fold_binders(
            &mut |ty, binders| match ty.kind(&Interner) {
                TyKind::BoundVar(bound) if bound.debruijn >= binders => {
                    TyKind::BoundVar(bound.shifted_out_to(n).unwrap_or(bound.clone()))
                        .intern(&Interner)
                }
                _ => ty,
            },
            DebruijnIndex::INNERMOST,
        )
    }
}

impl TypeWalk for Ty {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        match self.kind(&Interner) {
            TyKind::Alias(AliasTy::Projection(p_ty)) => {
                for t in p_ty.substitution.iter(&Interner) {
                    t.walk(f);
                }
            }
            TyKind::Alias(AliasTy::Opaque(o_ty)) => {
                for t in o_ty.substitution.iter(&Interner) {
                    t.walk(f);
                }
            }
            TyKind::Dyn(dyn_ty) => {
                for p in dyn_ty.bounds.value.interned().iter() {
                    p.walk(f);
                }
            }
            TyKind::Slice(ty) | TyKind::Array(ty) | TyKind::Ref(_, ty) | TyKind::Raw(_, ty) => {
                ty.walk(f);
            }
            _ => {
                if let Some(substs) = self.substs() {
                    for t in substs.iter(&Interner) {
                        t.walk(f);
                    }
                }
            }
        }
        f(self);
    }

    fn walk_mut_binders(
        &mut self,
        f: &mut impl FnMut(&mut Ty, DebruijnIndex),
        binders: DebruijnIndex,
    ) {
        match self.interned_mut() {
            TyKind::Alias(AliasTy::Projection(p_ty)) => {
                p_ty.substitution.walk_mut_binders(f, binders);
            }
            TyKind::Dyn(dyn_ty) => {
                for p in make_mut_slice(&mut dyn_ty.bounds.value.0) {
                    p.walk_mut_binders(f, binders.shifted_in());
                }
            }
            TyKind::Alias(AliasTy::Opaque(o_ty)) => {
                o_ty.substitution.walk_mut_binders(f, binders);
            }
            TyKind::Slice(ty) | TyKind::Array(ty) | TyKind::Ref(_, ty) | TyKind::Raw(_, ty) => {
                ty.walk_mut_binders(f, binders);
            }
            _ => {
                if let Some(substs) = self.substs_mut() {
                    substs.walk_mut_binders(f, binders);
                }
            }
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
pub enum ImplTraitId {
    ReturnTypeImplTrait(hir_def::FunctionId, u16),
    AsyncBlockTypeImplTrait(hir_def::DefWithBodyId, ExprId),
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct ReturnTypeImplTraits {
    pub(crate) impl_traits: Vec<ReturnTypeImplTrait>,
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub(crate) struct ReturnTypeImplTrait {
    pub(crate) bounds: Binders<Vec<QuantifiedWhereClause>>,
}

pub fn to_foreign_def_id(id: TypeAliasId) -> ForeignDefId {
    chalk_ir::ForeignDefId(salsa::InternKey::as_intern_id(&id))
}

pub fn from_foreign_def_id(id: ForeignDefId) -> TypeAliasId {
    salsa::InternKey::from_intern_id(id.0)
}

pub fn to_assoc_type_id(id: TypeAliasId) -> AssocTypeId {
    chalk_ir::AssocTypeId(salsa::InternKey::as_intern_id(&id))
}

pub fn from_assoc_type_id(id: AssocTypeId) -> TypeAliasId {
    salsa::InternKey::from_intern_id(id.0)
}

pub fn from_placeholder_idx(db: &dyn HirDatabase, idx: PlaceholderIndex) -> TypeParamId {
    assert_eq!(idx.ui, chalk_ir::UniverseIndex::ROOT);
    let interned_id = salsa::InternKey::from_intern_id(salsa::InternId::from(idx.idx));
    db.lookup_intern_type_param_id(interned_id)
}

pub fn to_placeholder_idx(db: &dyn HirDatabase, id: TypeParamId) -> PlaceholderIndex {
    let interned_id = db.intern_type_param_id(id);
    PlaceholderIndex {
        ui: chalk_ir::UniverseIndex::ROOT,
        idx: salsa::InternKey::as_intern_id(&interned_id).as_usize(),
    }
}

pub fn to_chalk_trait_id(id: TraitId) -> ChalkTraitId {
    chalk_ir::TraitId(salsa::InternKey::as_intern_id(&id))
}

pub fn from_chalk_trait_id(id: ChalkTraitId) -> TraitId {
    salsa::InternKey::from_intern_id(id.0)
}
