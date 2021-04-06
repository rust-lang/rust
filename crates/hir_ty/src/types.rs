//! This is the home of `Ty` etc. until they get replaced by their chalk_ir
//! equivalents.

use std::sync::Arc;

use chalk_ir::{
    cast::{Cast, CastTo, Caster},
    BoundVar, Mutability, Scalar, TyVariableKind,
};
use smallvec::SmallVec;

use crate::{
    AssocTypeId, CanonicalVarKinds, ChalkTraitId, ClosureId, Const, FnDefId, FnSig, ForeignDefId,
    Interner, Lifetime, OpaqueTyId, PlaceholderIndex, TypeWalk, VariableKind, VariableKinds,
};

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct OpaqueTy {
    pub opaque_ty_id: OpaqueTyId,
    pub substitution: Substitution,
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
    pub fn self_type_parameter(&self, interner: &Interner) -> &Ty {
        &self.substitution.interned()[0].assert_ty_ref(interner)
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct DynTy {
    /// The unknown self type.
    pub bounds: Binders<QuantifiedWhereClauses>,
    pub lifetime: Lifetime,
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct FnPointer {
    pub num_binders: usize,
    pub sig: FnSig,
    pub substitution: FnSubst,
}
/// A wrapper for the substs on a Fn.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct FnSubst(pub Substitution);

impl FnPointer {
    /// Represent the current `Fn` as if it was wrapped in `Binders`
    pub fn into_binders(self, interner: &Interner) -> Binders<FnSubst> {
        Binders::new(
            VariableKinds::from_iter(
                interner,
                (0..self.num_binders).map(|_| VariableKind::Lifetime),
            ),
            self.substitution,
        )
    }

    /// Represent the current `Fn` as if it was wrapped in `Binders`
    pub fn as_binders(&self, interner: &Interner) -> Binders<&FnSubst> {
        Binders::new(
            VariableKinds::from_iter(
                interner,
                (0..self.num_binders).map(|_| VariableKind::Lifetime),
            ),
            &self.substitution,
        )
    }
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

/// A type.
///
/// See also the `TyKind` enum in rustc (librustc/ty/sty.rs), which represents
/// the same thing (but in a different way).
///
/// This should be cheap to clone.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub enum TyKind {
    /// Structures, enumerations and unions.
    Adt(chalk_ir::AdtId<Interner>, Substitution),

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
    Array(Ty, Const),

    /// The pointee of an array slice.  Written as `[T]`.
    Slice(Ty),

    /// A raw pointer. Written as `*mut T` or `*const T`
    Raw(Mutability, Ty),

    /// A reference; a pointer with an associated lifetime. Written as
    /// `&'a mut T` or `&'a T`.
    Ref(Mutability, Lifetime, Ty),

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
    Foreign(ForeignDefId),

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
    Error,
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

    pub fn interned_mut(&mut self) -> &mut GenericArgData {
        &mut self.interned
    }
}

/// A list of substitutions for generic parameters.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct Substitution(SmallVec<[GenericArg; 2]>);

impl Substitution {
    pub fn interned(&self) -> &[GenericArg] {
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

    pub fn from1(_interner: &Interner, ty: Ty) -> Substitution {
        Substitution::intern({
            let mut v = SmallVec::new();
            v.push(ty.cast(&Interner));
            v
        })
    }

    pub fn from_iter(
        interner: &Interner,
        elements: impl IntoIterator<Item = impl CastTo<GenericArg>>,
    ) -> Self {
        Substitution(elements.into_iter().casted(interner).collect())
    }

    pub fn apply<T: TypeWalk>(&self, value: T, _interner: &Interner) -> T {
        value.subst_bound_vars(self)
    }

    // Temporary helper functions, to be removed
    pub fn intern(interned: SmallVec<[GenericArg; 2]>) -> Substitution {
        Substitution(interned)
    }

    pub fn interned_mut(&mut self) -> &mut SmallVec<[GenericArg; 2]> {
        &mut self.0
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Binders<T> {
    /// The binders that quantify over the value.
    pub binders: VariableKinds,
    value: T,
}

impl<T> Binders<T> {
    pub fn new(binders: VariableKinds, value: T) -> Self {
        Self { binders, value }
    }

    pub fn empty(_interner: &Interner, value: T) -> Self {
        crate::make_only_type_binders(0, value)
    }

    pub fn as_ref(&self) -> Binders<&T> {
        Binders { binders: self.binders.clone(), value: &self.value }
    }

    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Binders<U> {
        Binders { binders: self.binders, value: f(self.value) }
    }

    pub fn filter_map<U>(self, f: impl FnOnce(T) -> Option<U>) -> Option<Binders<U>> {
        Some(Binders { binders: self.binders, value: f(self.value)? })
    }

    pub fn skip_binders(&self) -> &T {
        &self.value
    }

    pub fn into_value_and_skipped_binders(self) -> (T, VariableKinds) {
        (self.value, self.binders)
    }

    /// Returns the number of binders.
    pub fn len(&self, interner: &Interner) -> usize {
        self.binders.len(interner)
    }

    // Temporary helper function, to be removed
    pub fn skip_binders_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

impl<T: Clone> Binders<&T> {
    pub fn cloned(&self) -> Binders<T> {
        Binders::new(self.binders.clone(), self.value.clone())
    }
}

impl<T: TypeWalk> Binders<T> {
    /// Substitutes all variables.
    pub fn substitute(self, interner: &Interner, subst: &Substitution) -> T {
        let (value, binders) = self.into_value_and_skipped_binders();
        assert_eq!(subst.len(interner), binders.len(interner));
        value.subst_bound_vars(subst)
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for Binders<T> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        let Binders { ref binders, ref value } = *self;
        write!(fmt, "for{:?} ", binders.inner_debug(&Interner))?;
        std::fmt::Debug::fmt(value, fmt)
    }
}

/// A trait with type parameters. This includes the `Self`, so this represents a concrete type implementing the trait.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct TraitRef {
    pub trait_id: ChalkTraitId,
    pub substitution: Substitution,
}

impl TraitRef {
    pub fn self_type_parameter(&self, interner: &Interner) -> &Ty {
        &self.substitution.at(interner, 0).assert_ty_ref(interner)
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

    pub fn interned_mut(&mut self) -> &mut Arc<[QuantifiedWhereClause]> {
        &mut self.0
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

/// Something (usually a goal), along with an environment.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct InEnvironment<T> {
    pub environment: chalk_ir::Environment<Interner>,
    pub goal: T,
}

impl<T> InEnvironment<T> {
    pub fn new(environment: chalk_ir::Environment<Interner>, value: T) -> InEnvironment<T> {
        InEnvironment { environment, goal: value }
    }
}

/// Something that needs to be proven (by Chalk) during type checking, e.g. that
/// a certain type implements a certain trait. Proving the Obligation might
/// result in additional information about inference variables.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DomainGoal {
    Holds(WhereClause),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AliasEq {
    pub alias: AliasTy,
    pub ty: Ty,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstrainedSubst {
    pub subst: Substitution,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// A (possible) solution for a proposed goal.
pub enum Solution {
    /// The goal indeed holds, and there is a unique value for all existential
    /// variables.
    Unique(Canonical<ConstrainedSubst>),

    /// The goal may be provable in multiple ways, but regardless we may have some guidance
    /// for type inference. In this case, we don't return any lifetime
    /// constraints, since we have not "committed" to any particular solution
    /// yet.
    Ambig(Guidance),
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// When a goal holds ambiguously (e.g., because there are multiple possible
/// solutions), we issue a set of *guidance* back to type inference.
pub enum Guidance {
    /// The existential variables *must* have the given values if the goal is
    /// ever to hold, but that alone isn't enough to guarantee the goal will
    /// actually hold.
    Definite(Canonical<Substitution>),

    /// There are multiple plausible values for the existentials, but the ones
    /// here are suggested as the preferred choice heuristically. These should
    /// be used for inference fallback only.
    Suggested(Canonical<Substitution>),

    /// There's no useful information to feed back to type inference
    Unknown,
}

/// The kinds of placeholders we need during type inference. There's separate
/// values for general types, and for integer and float variables. The latter
/// two are used for inference of literal values (e.g. `100` could be one of
/// several integer types).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct InferenceVar {
    index: u32,
}

impl From<u32> for InferenceVar {
    fn from(index: u32) -> InferenceVar {
        InferenceVar { index }
    }
}

impl InferenceVar {
    /// Gets the underlying index value.
    pub fn index(self) -> u32 {
        self.index
    }
}
