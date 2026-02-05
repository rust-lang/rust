//! Set of traits which are used to emulate the inherent impls that are present in `rustc_middle`.
//! It is customary to glob-import `rustc_type_ir::inherent::*` to bring all of these traits into
//! scope when programming in interner-agnostic settings, and to avoid importing any of these
//! directly elsewhere (i.e. specify the full path for an implementation downstream).

use core::ops::Range;
use std::fmt::Debug;
use std::hash::Hash;

use crate::elaborate::Elaboratable;
use crate::fold::{TypeFoldable, TypeSuperFoldable};
use crate::relate::Relate;
use crate::solve::{AdtDestructorKind, SizedTraitKind};
use crate::visit::{Flags, TypeSuperVisitable, TypeVisitable};
use crate::{self as ty, ClauseKind, FieldInfo, Interner, PredicateKind, Ty, UpcastFrom};

pub trait Tys<I: Interner<Tys = Self>>:
    Copy + Debug + Hash + Eq + SliceLike<Item = Ty<I>> + TypeFoldable<I> + Default
{
    fn inputs(self) -> I::FnInputTys;

    fn output(self) -> Ty<I>;
}

pub trait Abi<I: Interner<Abi = Self>>: Copy + Debug + Hash + Eq {
    fn rust() -> Self;

    /// Whether this ABI is `extern "Rust"`.
    fn is_rust(self) -> bool;
}

pub trait Safety<I: Interner<Safety = Self>>: Copy + Debug + Hash + Eq {
    fn safe() -> Self;

    fn is_safe(self) -> bool;

    fn prefix_str(self) -> &'static str;
}

pub trait Region<I: Interner<Region = Self>>:
    Copy
    + Debug
    + Hash
    + Eq
    + Into<I::GenericArg>
    + IntoKind<Kind = ty::RegionKind<I>>
    + Flags
    + Relate<I>
{
    fn new_bound(interner: I, debruijn: ty::DebruijnIndex, var: ty::BoundRegion<I>) -> Self;

    fn new_anon_bound(interner: I, debruijn: ty::DebruijnIndex, var: ty::BoundVar) -> Self;

    fn new_canonical_bound(interner: I, var: ty::BoundVar) -> Self;

    fn new_static(interner: I) -> Self;

    fn new_placeholder(interner: I, var: ty::PlaceholderRegion<I>) -> Self;

    fn is_bound(self) -> bool {
        matches!(self.kind(), ty::ReBound(..))
    }
}

pub trait Const<I: Interner<Const = Self>>:
    Copy
    + Debug
    + Hash
    + Eq
    + Into<I::GenericArg>
    + Into<I::Term>
    + IntoKind<Kind = ty::ConstKind<I>>
    + TypeSuperVisitable<I>
    + TypeSuperFoldable<I>
    + Relate<I>
    + Flags
{
    fn new_infer(interner: I, var: ty::InferConst) -> Self;

    fn new_var(interner: I, var: ty::ConstVid) -> Self;

    fn new_bound(interner: I, debruijn: ty::DebruijnIndex, bound_const: ty::BoundConst<I>) -> Self;

    fn new_anon_bound(interner: I, debruijn: ty::DebruijnIndex, var: ty::BoundVar) -> Self;

    fn new_canonical_bound(interner: I, var: ty::BoundVar) -> Self;

    fn new_placeholder(interner: I, param: ty::PlaceholderConst<I>) -> Self;

    fn new_unevaluated(interner: I, uv: ty::UnevaluatedConst<I>) -> Self;

    fn new_expr(interner: I, expr: I::ExprConst) -> Self;

    fn new_error(interner: I, guar: I::ErrorGuaranteed) -> Self;

    fn new_error_with_message(interner: I, msg: impl ToString) -> Self {
        Self::new_error(interner, interner.delay_bug(msg))
    }

    fn is_ct_var(self) -> bool {
        matches!(self.kind(), ty::ConstKind::Infer(ty::InferConst::Var(_)))
    }

    fn is_ct_error(self) -> bool {
        matches!(self.kind(), ty::ConstKind::Error(_))
    }

    fn try_to_target_usize(&self, interner: I) -> Option<u64>;

    fn from_target_usize(interner: I, n: u64) -> Self;
}

pub trait ValueConst<I: Interner<ValueConst = Self>>: Copy + Debug + Hash + Eq {
    fn ty(self) -> Ty<I>;
    fn valtree(self) -> I::ValTree;
}

pub trait ExprConst<I: Interner<ExprConst = Self>>: Copy + Debug + Hash + Eq + Relate<I> {
    fn args(self) -> I::GenericArgs;
}

pub trait GenericsOf<I: Interner<GenericsOf = Self>> {
    fn count(&self) -> usize;
}

pub trait GenericArg<I: Interner<GenericArg = Self>>:
    Copy
    + Debug
    + Hash
    + Eq
    + IntoKind<Kind = ty::GenericArgKind<I>>
    + TypeVisitable<I>
    + Relate<I>
    + From<Ty<I>>
    + From<I::Region>
    + From<I::Const>
    + From<I::Term>
{
    fn as_term(&self) -> Option<I::Term> {
        match self.kind() {
            ty::GenericArgKind::Lifetime(_) => None,
            ty::GenericArgKind::Type(ty) => Some(ty.into()),
            ty::GenericArgKind::Const(ct) => Some(ct.into()),
        }
    }

    fn as_type(&self) -> Option<Ty<I>> {
        if let ty::GenericArgKind::Type(ty) = self.kind() { Some(ty) } else { None }
    }

    fn expect_ty(&self) -> Ty<I> {
        self.as_type().expect("expected a type")
    }

    fn as_const(&self) -> Option<I::Const> {
        if let ty::GenericArgKind::Const(c) = self.kind() { Some(c) } else { None }
    }

    fn expect_const(&self) -> I::Const {
        self.as_const().expect("expected a const")
    }

    fn as_region(&self) -> Option<I::Region> {
        if let ty::GenericArgKind::Lifetime(c) = self.kind() { Some(c) } else { None }
    }

    fn expect_region(&self) -> I::Region {
        self.as_region().expect("expected a const")
    }

    fn is_non_region_infer(self) -> bool {
        match self.kind() {
            ty::GenericArgKind::Lifetime(_) => false,
            ty::GenericArgKind::Type(ty) => ty.is_ty_var(),
            ty::GenericArgKind::Const(ct) => ct.is_ct_var(),
        }
    }
}

pub trait Term<I: Interner<Term = Self>>:
    Copy
    + Debug
    + Hash
    + Eq
    + From<Ty<I>>
    + IntoKind<Kind = ty::TermKind<I>>
    + TypeFoldable<I>
    + Relate<I>
{
    fn as_type(&self) -> Option<Ty<I>> {
        if let ty::TermKind::Ty(ty) = self.kind() { Some(ty) } else { None }
    }

    fn expect_ty(&self) -> Ty<I> {
        self.as_type().expect("expected a type, but found a const")
    }

    fn as_const(&self) -> Option<I::Const> {
        if let ty::TermKind::Const(c) = self.kind() { Some(c) } else { None }
    }

    fn expect_const(&self) -> I::Const {
        self.as_const().expect("expected a const, but found a type")
    }

    fn is_infer(self) -> bool {
        match self.kind() {
            ty::TermKind::Ty(ty) => ty.is_ty_var(),
            ty::TermKind::Const(ct) => ct.is_ct_var(),
        }
    }

    fn is_error(self) -> bool {
        match self.kind() {
            ty::TermKind::Ty(ty) => ty.is_ty_error(),
            ty::TermKind::Const(ct) => ct.is_ct_error(),
        }
    }

    fn to_alias_term(self) -> Option<ty::AliasTerm<I>> {
        match self.kind() {
            ty::TermKind::Ty(ty) => match ty.kind() {
                ty::Alias(_kind, alias_ty) => Some(alias_ty.into()),
                _ => None,
            },
            ty::TermKind::Const(ct) => match ct.kind() {
                ty::ConstKind::Unevaluated(uv) => Some(uv.into()),
                _ => None,
            },
        }
    }
}

pub trait GenericArgs<I: Interner<GenericArgs = Self>>:
    Copy + Debug + Hash + Eq + SliceLike<Item = I::GenericArg> + Default + Relate<I>
{
    fn rebase_onto(
        self,
        interner: I,
        source_def_id: I::DefId,
        target: I::GenericArgs,
    ) -> I::GenericArgs;

    fn type_at(self, i: usize) -> Ty<I>;

    fn region_at(self, i: usize) -> I::Region;

    fn const_at(self, i: usize) -> I::Const;

    fn identity_for_item(interner: I, def_id: I::DefId) -> I::GenericArgs;

    fn extend_with_error(
        interner: I,
        def_id: I::DefId,
        original_args: &[I::GenericArg],
    ) -> I::GenericArgs;

    fn split_closure_args(self) -> ty::ClosureArgsParts<I>;
    fn split_coroutine_closure_args(self) -> ty::CoroutineClosureArgsParts<I>;
    fn split_coroutine_args(self) -> ty::CoroutineArgsParts<I>;

    fn as_closure(self) -> ty::ClosureArgs<I> {
        ty::ClosureArgs { args: self }
    }
    fn as_coroutine_closure(self) -> ty::CoroutineClosureArgs<I> {
        ty::CoroutineClosureArgs { args: self }
    }
    fn as_coroutine(self) -> ty::CoroutineArgs<I> {
        ty::CoroutineArgs { args: self }
    }

    fn coroutine_discriminant_for_variant(
        self,
        def_id: I::DefId,
        interner: I,
        variant_index: I::VariantIdx,
    ) -> I::Discr;

    fn as_coroutine_discr_ty(self, interner: I) -> Ty<I>;
}

pub trait Predicate<I: Interner<Predicate = Self>>:
    Copy
    + Debug
    + Hash
    + Eq
    + TypeSuperVisitable<I>
    + TypeSuperFoldable<I>
    + Flags
    + UpcastFrom<I, ty::PredicateKind<I>>
    + UpcastFrom<I, ty::Binder<I, ty::PredicateKind<I>>>
    + UpcastFrom<I, ty::ClauseKind<I>>
    + UpcastFrom<I, ty::Binder<I, ty::ClauseKind<I>>>
    + UpcastFrom<I, I::Clause>
    + UpcastFrom<I, ty::NormalizesTo<I>>
    + UpcastFrom<I, ty::TraitRef<I>>
    + UpcastFrom<I, ty::Binder<I, ty::TraitRef<I>>>
    + UpcastFrom<I, ty::TraitPredicate<I>>
    + UpcastFrom<I, ty::OutlivesPredicate<I, Ty<I>>>
    + UpcastFrom<I, ty::OutlivesPredicate<I, I::Region>>
    + IntoKind<Kind = ty::Binder<I, ty::PredicateKind<I>>>
    + Elaboratable<I>
{
    fn as_clause(self) -> Option<I::Clause>;

    fn as_normalizes_to(self) -> Option<ty::Binder<I, ty::NormalizesTo<I>>> {
        let kind = self.kind();
        match kind.skip_binder() {
            ty::PredicateKind::NormalizesTo(pred) => Some(kind.rebind(pred)),
            _ => None,
        }
    }

    fn allow_normalization(self) -> bool {
        match self.kind().skip_binder() {
            PredicateKind::Clause(ClauseKind::WellFormed(_)) | PredicateKind::AliasRelate(..) => {
                false
            }
            PredicateKind::Clause(ClauseKind::Trait(_))
            | PredicateKind::Clause(ClauseKind::HostEffect(..))
            | PredicateKind::Clause(ClauseKind::RegionOutlives(_))
            | PredicateKind::Clause(ClauseKind::TypeOutlives(_))
            | PredicateKind::Clause(ClauseKind::Projection(_))
            | PredicateKind::Clause(ClauseKind::ConstArgHasType(..))
            | PredicateKind::Clause(ClauseKind::UnstableFeature(_))
            | PredicateKind::DynCompatible(_)
            | PredicateKind::Subtype(_)
            | PredicateKind::Coerce(_)
            | PredicateKind::Clause(ClauseKind::ConstEvaluatable(_))
            | PredicateKind::ConstEquate(_, _)
            | PredicateKind::NormalizesTo(..)
            | PredicateKind::Ambiguous => true,
        }
    }
}

pub trait Clause<I: Interner<Clause = Self>>:
    Copy
    + Debug
    + Hash
    + Eq
    + TypeFoldable<I>
    + UpcastFrom<I, ty::Binder<I, ty::ClauseKind<I>>>
    + UpcastFrom<I, ty::TraitRef<I>>
    + UpcastFrom<I, ty::Binder<I, ty::TraitRef<I>>>
    + UpcastFrom<I, ty::TraitPredicate<I>>
    + UpcastFrom<I, ty::Binder<I, ty::TraitPredicate<I>>>
    + UpcastFrom<I, ty::ProjectionPredicate<I>>
    + UpcastFrom<I, ty::Binder<I, ty::ProjectionPredicate<I>>>
    + IntoKind<Kind = ty::Binder<I, ty::ClauseKind<I>>>
    + Elaboratable<I>
{
    fn as_predicate(self) -> I::Predicate;

    fn as_trait_clause(self) -> Option<ty::Binder<I, ty::TraitPredicate<I>>> {
        self.kind()
            .map_bound(|clause| if let ty::ClauseKind::Trait(t) = clause { Some(t) } else { None })
            .transpose()
    }

    fn as_host_effect_clause(self) -> Option<ty::Binder<I, ty::HostEffectPredicate<I>>> {
        self.kind()
            .map_bound(
                |clause| if let ty::ClauseKind::HostEffect(t) = clause { Some(t) } else { None },
            )
            .transpose()
    }

    fn as_projection_clause(self) -> Option<ty::Binder<I, ty::ProjectionPredicate<I>>> {
        self.kind()
            .map_bound(
                |clause| {
                    if let ty::ClauseKind::Projection(p) = clause { Some(p) } else { None }
                },
            )
            .transpose()
    }

    /// Performs a instantiation suitable for going from a
    /// poly-trait-ref to supertraits that must hold if that
    /// poly-trait-ref holds. This is slightly different from a normal
    /// instantiation in terms of what happens with bound regions.
    fn instantiate_supertrait(self, cx: I, trait_ref: ty::Binder<I, ty::TraitRef<I>>) -> Self;
}

pub trait Clauses<I: Interner<Clauses = Self>>:
    Copy
    + Debug
    + Hash
    + Eq
    + TypeSuperVisitable<I>
    + TypeSuperFoldable<I>
    + Flags
    + SliceLike<Item = I::Clause>
{
}

pub trait IntoKind {
    type Kind;

    fn kind(self) -> Self::Kind;
}

pub trait ParamLike: Copy + Debug + Hash + Eq {
    fn index(self) -> u32;
}

pub trait AdtDef<I: Interner>: Copy + Debug + Hash + Eq {
    fn def_id(self) -> I::AdtId;

    fn is_struct(self) -> bool;

    fn is_packed(self) -> bool;

    fn is_box(self) -> bool;

    fn is_pin(self) -> bool;

    fn is_enum(self) -> bool;

    fn is_union(self) -> bool;

    /// Returns the type of the struct tail.
    ///
    /// Expects the `AdtDef` to be a struct. If it is not, then this will panic.
    fn struct_tail_ty(self, interner: I) -> Option<ty::EarlyBinder<I, Ty<I>>>;

    fn is_phantom_data(self) -> bool;

    fn is_manually_drop(self) -> bool;

    fn field_representing_type_info(
        self,
        interner: I,
        args: I::GenericArgs,
    ) -> Option<FieldInfo<I>>;

    fn has_unsafe_fields(self) -> bool;

    // FIXME: perhaps use `all_fields` and expose `FieldDef`.
    fn all_field_tys(self, interner: I) -> ty::EarlyBinder<I, impl IntoIterator<Item = Ty<I>>>;

    fn sizedness_constraint(
        self,
        interner: I,
        sizedness: SizedTraitKind,
    ) -> Option<ty::EarlyBinder<I, Ty<I>>>;

    fn is_fundamental(self) -> bool;

    fn destructor(self, interner: I) -> Option<AdtDestructorKind>;

    fn non_enum_variant(self) -> I::VariantDef;

    fn repr_is_simd(self) -> bool;

    fn scalable_element_cnt(self) -> Option<u16>;

    fn variant_range(self) -> Range<I::VariantIdx>;

    fn discriminant_for_variant(self, interner: I, variant_index: I::VariantIdx) -> I::Discr;

    fn repr_discr_type_to_ty(self, interner: I) -> Ty<I>;
}

pub trait VariantDef<I: Interner>: Debug + Hash {
    fn field_zero_ty(self, interner: I, args: I::GenericArgs) -> Ty<I>;

    fn fields_len(self) -> usize;
}

pub trait ParamEnv<I: Interner>: Copy + Debug + Hash + Eq + TypeFoldable<I> {
    fn caller_bounds(self) -> impl SliceLike<Item = I::Clause>;
}

pub trait Features<I: Interner>: Copy {
    fn generic_const_exprs(self) -> bool;

    fn coroutine_clone(self) -> bool;

    fn feature_bound_holds_in_crate(self, symbol: I::Symbol) -> bool;
}

pub trait DefId<I: Interner>: Copy + Debug + Hash + Eq + TypeFoldable<I> {
    fn is_local(self) -> bool;

    fn as_local(self) -> Option<I::LocalDefId>;
}

pub trait SpecificDefId<I: Interner>:
    DefId<I> + Into<I::DefId> + TryFrom<I::DefId, Error: std::fmt::Debug>
{
}

impl<I: Interner, T: DefId<I> + Into<I::DefId> + TryFrom<I::DefId, Error: std::fmt::Debug>>
    SpecificDefId<I> for T
{
}

pub trait BoundExistentialPredicates<I: Interner>:
    Copy + Debug + Hash + Eq + Relate<I> + SliceLike<Item = ty::Binder<I, ty::ExistentialPredicate<I>>>
{
    fn principal_def_id(self) -> Option<I::TraitId>;

    fn principal(self) -> Option<ty::Binder<I, ty::ExistentialTraitRef<I>>>;

    fn auto_traits(self) -> impl IntoIterator<Item = I::TraitId>;

    fn projection_bounds(
        self,
    ) -> impl IntoIterator<Item = ty::Binder<I, ty::ExistentialProjection<I>>>;
}

pub trait Span<I: Interner>: Copy + Debug + Hash + Eq + TypeFoldable<I> {
    fn dummy() -> Self;
}

pub trait OpaqueTypeStorageEntries: Debug + Copy + Default {
    /// Whether the number of opaques has changed in a way that necessitates
    /// reevaluating a goal. For now, this is only when the number of non-duplicated
    /// entries changed.
    fn needs_reevaluation(self, canonicalized: usize) -> bool;
}

pub trait SliceLike: Sized + Copy {
    type Item: Copy;
    type IntoIter: Iterator<Item = Self::Item> + DoubleEndedIterator;

    fn iter(self) -> Self::IntoIter;

    fn as_slice(&self) -> &[Self::Item];

    fn get(self, idx: usize) -> Option<Self::Item> {
        self.as_slice().get(idx).copied()
    }

    fn len(self) -> usize {
        self.as_slice().len()
    }

    fn is_empty(self) -> bool {
        self.len() == 0
    }

    fn contains(self, t: &Self::Item) -> bool
    where
        Self::Item: PartialEq,
    {
        self.as_slice().contains(t)
    }

    fn to_vec(self) -> Vec<Self::Item> {
        self.as_slice().to_vec()
    }

    fn last(self) -> Option<Self::Item> {
        self.as_slice().last().copied()
    }

    fn split_last(&self) -> Option<(&Self::Item, &[Self::Item])> {
        self.as_slice().split_last()
    }

    fn first(&self) -> Option<&Self::Item> {
        self.as_slice().first()
    }
}

impl<'a, T: Copy> SliceLike for &'a [T] {
    type Item = T;
    type IntoIter = std::iter::Copied<std::slice::Iter<'a, T>>;

    fn iter(self) -> Self::IntoIter {
        self.iter().copied()
    }

    fn as_slice(&self) -> &[Self::Item] {
        *self
    }
}

impl<'a, T: Copy, const N: usize> SliceLike for &'a [T; N] {
    type Item = T;
    type IntoIter = std::iter::Copied<std::slice::Iter<'a, T>>;

    fn iter(self) -> Self::IntoIter {
        self.into_iter().copied()
    }

    fn as_slice(&self) -> &[Self::Item] {
        *self
    }
}

impl<'a, S: SliceLike> SliceLike for &'a S {
    type Item = S::Item;
    type IntoIter = S::IntoIter;

    fn iter(self) -> Self::IntoIter {
        (*self).iter()
    }

    fn as_slice(&self) -> &[Self::Item] {
        (*self).as_slice()
    }
}

pub trait Symbol<I>: Copy + Hash + PartialEq + Eq + Debug {
    fn is_kw_underscore_lifetime(self) -> bool;
}

pub trait Interned<T>: Clone + Copy + Hash + PartialEq + Eq + Debug {
    fn new_unchecked(t: &T) -> Self;
}
