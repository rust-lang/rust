//! Set of traits which are used to emulate the inherent impls that are present in `rustc_middle`.
//! It is customary to glob-import `rustc_type_ir::inherent::*` to bring all of these traits into
//! scope when programming in interner-agnostic settings, and to avoid importing any of these
//! directly elsewhere (i.e. specify the full path for an implementation downstream).

use std::fmt::Debug;
use std::hash::Hash;

use rustc_ast_ir::Mutability;

use crate::elaborate::Elaboratable;
use crate::fold::{TypeFoldable, TypeSuperFoldable};
use crate::relate::Relate;
use crate::solve::AdtDestructorKind;
use crate::visit::{Flags, TypeSuperVisitable, TypeVisitable};
use crate::{self as ty, CollectAndApply, Interner, UpcastFrom};

pub trait Ty<I: Interner<Ty = Self>>:
    Copy
    + Debug
    + Hash
    + Eq
    + Into<I::GenericArg>
    + Into<I::Term>
    + IntoKind<Kind = ty::TyKind<I>>
    + TypeSuperVisitable<I>
    + TypeSuperFoldable<I>
    + Relate<I>
    + Flags
{
    fn new_unit(interner: I) -> Self;

    fn new_bool(interner: I) -> Self;

    fn new_u8(interner: I) -> Self;

    fn new_usize(interner: I) -> Self;

    fn new_infer(interner: I, var: ty::InferTy) -> Self;

    fn new_var(interner: I, var: ty::TyVid) -> Self;

    fn new_param(interner: I, param: I::ParamTy) -> Self;

    fn new_placeholder(interner: I, param: I::PlaceholderTy) -> Self;

    fn new_bound(interner: I, debruijn: ty::DebruijnIndex, var: I::BoundTy) -> Self;

    fn new_anon_bound(interner: I, debruijn: ty::DebruijnIndex, var: ty::BoundVar) -> Self;

    fn new_alias(interner: I, kind: ty::AliasTyKind, alias_ty: ty::AliasTy<I>) -> Self;

    fn new_projection_from_args(interner: I, def_id: I::DefId, args: I::GenericArgs) -> Self {
        Ty::new_alias(
            interner,
            ty::AliasTyKind::Projection,
            ty::AliasTy::new_from_args(interner, def_id, args),
        )
    }

    fn new_projection(
        interner: I,
        def_id: I::DefId,
        args: impl IntoIterator<Item: Into<I::GenericArg>>,
    ) -> Self {
        Ty::new_alias(
            interner,
            ty::AliasTyKind::Projection,
            ty::AliasTy::new(interner, def_id, args),
        )
    }

    fn new_error(interner: I, guar: I::ErrorGuaranteed) -> Self;

    fn new_adt(interner: I, adt_def: I::AdtDef, args: I::GenericArgs) -> Self;

    fn new_foreign(interner: I, def_id: I::DefId) -> Self;

    fn new_dynamic(
        interner: I,
        preds: I::BoundExistentialPredicates,
        region: I::Region,
        kind: ty::DynKind,
    ) -> Self;

    fn new_coroutine(interner: I, def_id: I::DefId, args: I::GenericArgs) -> Self;

    fn new_coroutine_closure(interner: I, def_id: I::DefId, args: I::GenericArgs) -> Self;

    fn new_closure(interner: I, def_id: I::DefId, args: I::GenericArgs) -> Self;

    fn new_coroutine_witness(interner: I, def_id: I::DefId, args: I::GenericArgs) -> Self;

    fn new_ptr(interner: I, ty: Self, mutbl: Mutability) -> Self;

    fn new_ref(interner: I, region: I::Region, ty: Self, mutbl: Mutability) -> Self;

    fn new_array_with_const_len(interner: I, ty: Self, len: I::Const) -> Self;

    fn new_slice(interner: I, ty: Self) -> Self;

    fn new_tup(interner: I, tys: &[I::Ty]) -> Self;

    fn new_tup_from_iter<It, T>(interner: I, iter: It) -> T::Output
    where
        It: Iterator<Item = T>,
        T: CollectAndApply<Self, Self>;

    fn new_fn_def(interner: I, def_id: I::DefId, args: I::GenericArgs) -> Self;

    fn new_fn_ptr(interner: I, sig: ty::Binder<I, ty::FnSig<I>>) -> Self;

    fn new_pat(interner: I, ty: Self, pat: I::Pat) -> Self;

    fn tuple_fields(self) -> I::Tys;

    fn to_opt_closure_kind(self) -> Option<ty::ClosureKind>;

    fn from_closure_kind(interner: I, kind: ty::ClosureKind) -> Self;

    fn from_coroutine_closure_kind(interner: I, kind: ty::ClosureKind) -> Self;

    fn is_ty_var(self) -> bool {
        matches!(self.kind(), ty::Infer(ty::TyVar(_)))
    }

    fn is_floating_point(self) -> bool {
        matches!(self.kind(), ty::Float(_) | ty::Infer(ty::FloatVar(_)))
    }

    fn is_integral(self) -> bool {
        matches!(self.kind(), ty::Infer(ty::IntVar(_)) | ty::Int(_) | ty::Uint(_))
    }

    fn is_fn_ptr(self) -> bool {
        matches!(self.kind(), ty::FnPtr(..))
    }

    /// Checks whether this type is an ADT that has unsafe fields.
    fn has_unsafe_fields(self) -> bool;

    fn fn_sig(self, interner: I) -> ty::Binder<I, ty::FnSig<I>> {
        match self.kind() {
            ty::FnPtr(sig_tys, hdr) => sig_tys.with(hdr),
            ty::FnDef(def_id, args) => interner.fn_sig(def_id).instantiate(interner, args),
            ty::Error(_) => {
                // ignore errors (#54954)
                ty::Binder::dummy(ty::FnSig {
                    inputs_and_output: Default::default(),
                    c_variadic: false,
                    safety: I::Safety::safe(),
                    abi: I::Abi::rust(),
                })
            }
            ty::Closure(..) => panic!(
                "to get the signature of a closure, use `args.as_closure().sig()` not `fn_sig()`",
            ),
            _ => panic!("Ty::fn_sig() called on non-fn type: {:?}", self),
        }
    }

    fn discriminant_ty(self, interner: I) -> I::Ty;

    fn async_destructor_ty(self, interner: I) -> I::Ty;

    /// Returns `true` when the outermost type cannot be further normalized,
    /// resolved, or instantiated. This includes all primitive types, but also
    /// things like ADTs and trait objects, since even if their arguments or
    /// nested types may be further simplified, the outermost [`ty::TyKind`] or
    /// type constructor remains the same.
    fn is_known_rigid(self) -> bool {
        match self.kind() {
            ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Adt(_, _)
            | ty::Foreign(_)
            | ty::Str
            | ty::Array(_, _)
            | ty::Pat(_, _)
            | ty::Slice(_)
            | ty::RawPtr(_, _)
            | ty::Ref(_, _, _)
            | ty::FnDef(_, _)
            | ty::FnPtr(..)
            | ty::Dynamic(_, _, _)
            | ty::Closure(_, _)
            | ty::CoroutineClosure(_, _)
            | ty::Coroutine(_, _)
            | ty::CoroutineWitness(..)
            | ty::Never
            | ty::Tuple(_) => true,

            ty::Error(_)
            | ty::Infer(_)
            | ty::Alias(_, _)
            | ty::Param(_)
            | ty::Bound(_, _)
            | ty::Placeholder(_) => false,
        }
    }
}

pub trait Tys<I: Interner<Tys = Self>>:
    Copy + Debug + Hash + Eq + SliceLike<Item = I::Ty> + TypeFoldable<I> + Default
{
    fn inputs(self) -> I::FnInputTys;

    fn output(self) -> I::Ty;
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
    fn new_bound(interner: I, debruijn: ty::DebruijnIndex, var: I::BoundRegion) -> Self;

    fn new_anon_bound(interner: I, debruijn: ty::DebruijnIndex, var: ty::BoundVar) -> Self;

    fn new_static(interner: I) -> Self;

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

    fn new_bound(interner: I, debruijn: ty::DebruijnIndex, var: I::BoundConst) -> Self;

    fn new_anon_bound(interner: I, debruijn: ty::DebruijnIndex, var: ty::BoundVar) -> Self;

    fn new_unevaluated(interner: I, uv: ty::UnevaluatedConst<I>) -> Self;

    fn new_expr(interner: I, expr: I::ExprConst) -> Self;

    fn new_error(interner: I, guar: I::ErrorGuaranteed) -> Self;

    fn new_error_with_message(interner: I, msg: impl ToString) -> Self {
        Self::new_error(interner, interner.delay_bug(msg))
    }

    fn is_ct_var(self) -> bool {
        matches!(self.kind(), ty::ConstKind::Infer(ty::InferConst::Var(_)))
    }
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
    + From<I::Ty>
    + From<I::Region>
    + From<I::Const>
{
    fn as_type(&self) -> Option<I::Ty> {
        if let ty::GenericArgKind::Type(ty) = self.kind() { Some(ty) } else { None }
    }

    fn expect_ty(&self) -> I::Ty {
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
    Copy + Debug + Hash + Eq + IntoKind<Kind = ty::TermKind<I>> + TypeFoldable<I> + Relate<I>
{
    fn as_type(&self) -> Option<I::Ty> {
        if let ty::TermKind::Ty(ty) = self.kind() { Some(ty) } else { None }
    }

    fn expect_ty(&self) -> I::Ty {
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

    fn type_at(self, i: usize) -> I::Ty;

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
    + UpcastFrom<I, ty::OutlivesPredicate<I, I::Ty>>
    + UpcastFrom<I, ty::OutlivesPredicate<I, I::Region>>
    + IntoKind<Kind = ty::Binder<I, ty::PredicateKind<I>>>
    + Elaboratable<I>
{
    fn as_clause(self) -> Option<I::Clause>;

    fn is_coinductive(self, interner: I) -> bool;

    // FIXME: Eventually uplift the impl out of rustc and make this defaulted.
    fn allow_normalization(self) -> bool;
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

/// Common capabilities of placeholder kinds
pub trait PlaceholderLike: Copy + Debug + Hash + Eq {
    fn universe(self) -> ty::UniverseIndex;
    fn var(self) -> ty::BoundVar;

    fn with_updated_universe(self, ui: ty::UniverseIndex) -> Self;

    fn new(ui: ty::UniverseIndex, var: ty::BoundVar) -> Self;
}

pub trait IntoKind {
    type Kind;

    fn kind(self) -> Self::Kind;
}

pub trait BoundVarLike<I: Interner> {
    fn var(self) -> ty::BoundVar;

    fn assert_eq(self, var: I::BoundVarKind);
}

pub trait ParamLike {
    fn index(self) -> u32;
}

pub trait AdtDef<I: Interner>: Copy + Debug + Hash + Eq {
    fn def_id(self) -> I::DefId;

    fn is_struct(self) -> bool;

    /// Returns the type of the struct tail.
    ///
    /// Expects the `AdtDef` to be a struct. If it is not, then this will panic.
    fn struct_tail_ty(self, interner: I) -> Option<ty::EarlyBinder<I, I::Ty>>;

    fn is_phantom_data(self) -> bool;

    // FIXME: perhaps use `all_fields` and expose `FieldDef`.
    fn all_field_tys(self, interner: I) -> ty::EarlyBinder<I, impl IntoIterator<Item = I::Ty>>;

    fn sized_constraint(self, interner: I) -> Option<ty::EarlyBinder<I, I::Ty>>;

    fn is_fundamental(self) -> bool;

    fn destructor(self, interner: I) -> Option<AdtDestructorKind>;
}

pub trait ParamEnv<I: Interner>: Copy + Debug + Hash + Eq + TypeFoldable<I> {
    fn caller_bounds(self) -> impl SliceLike<Item = I::Clause>;
}

pub trait Features<I: Interner>: Copy {
    fn generic_const_exprs(self) -> bool;

    fn coroutine_clone(self) -> bool;

    fn associated_const_equality(self) -> bool;
}

pub trait DefId<I: Interner>: Copy + Debug + Hash + Eq + TypeFoldable<I> {
    fn is_local(self) -> bool;

    fn as_local(self) -> Option<I::LocalDefId>;
}

pub trait BoundExistentialPredicates<I: Interner>:
    Copy + Debug + Hash + Eq + Relate<I> + SliceLike<Item = ty::Binder<I, ty::ExistentialPredicate<I>>>
{
    fn principal_def_id(self) -> Option<I::DefId>;

    fn principal(self) -> Option<ty::Binder<I, ty::ExistentialTraitRef<I>>>;

    fn auto_traits(self) -> impl IntoIterator<Item = I::DefId>;

    fn projection_bounds(
        self,
    ) -> impl IntoIterator<Item = ty::Binder<I, ty::ExistentialProjection<I>>>;
}

pub trait Span<I: Interner>: Copy + Debug + Hash + Eq + TypeFoldable<I> {
    fn dummy() -> Self;
}

pub trait SliceLike: Sized + Copy {
    type Item: Copy;
    type IntoIter: Iterator<Item = Self::Item>;

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
