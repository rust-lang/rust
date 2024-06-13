//! Set of traits which are used to emulate the inherent impls that are present in `rustc_middle`.
//! It is customary to glob-import `rustc_type_ir::inherent::*` to bring all of these traits into
//! scope when programming in interner-agnostic settings, and to avoid importing any of these
//! directly elsewhere (i.e. specify the full path for an implementation downstream).

use std::fmt::Debug;
use std::hash::Hash;
use std::ops::Deref;

use rustc_ast_ir::Mutability;

use crate::fold::{TypeFoldable, TypeSuperFoldable};
use crate::relate::Relate;
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
    fn new_bool(interner: I) -> Self;

    fn new_u8(interner: I) -> Self;

    fn new_infer(interner: I, var: ty::InferTy) -> Self;

    fn new_var(interner: I, var: ty::TyVid) -> Self;

    fn new_bound(interner: I, debruijn: ty::DebruijnIndex, var: I::BoundTy) -> Self;

    fn new_anon_bound(interner: I, debruijn: ty::DebruijnIndex, var: ty::BoundVar) -> Self;

    fn new_alias(interner: I, kind: ty::AliasTyKind, alias_ty: ty::AliasTy<I>) -> Self;

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

    fn fn_sig(self, interner: I) -> ty::Binder<I, ty::FnSig<I>> {
        match self.kind() {
            ty::FnPtr(sig) => sig,
            ty::FnDef(def_id, args) => interner.fn_sig(def_id).instantiate(interner, &args),
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
}

pub trait Tys<I: Interner<Tys = Self>>:
    Copy
    + Debug
    + Hash
    + Eq
    + IntoIterator<Item = I::Ty>
    + Deref<Target: Deref<Target = [I::Ty]>>
    + TypeFoldable<I>
    + Default
{
    fn split_inputs_and_output(self) -> (I::FnInputTys, I::Ty);
}

pub trait Abi<I: Interner<Abi = Self>>: Copy + Debug + Hash + Eq + Relate<I> {
    fn rust() -> Self;

    /// Whether this ABI is `extern "Rust"`.
    fn is_rust(self) -> bool;
}

pub trait Safety<I: Interner<Safety = Self>>: Copy + Debug + Hash + Eq + Relate<I> {
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
    fn try_to_target_usize(self, interner: I) -> Option<u64>;

    fn new_infer(interner: I, var: ty::InferConst) -> Self;

    fn new_var(interner: I, var: ty::ConstVid) -> Self;

    fn new_bound(interner: I, debruijn: ty::DebruijnIndex, var: I::BoundConst) -> Self;

    fn new_anon_bound(interner: I, debruijn: ty::DebruijnIndex, var: ty::BoundVar) -> Self;

    fn new_unevaluated(interner: I, uv: ty::UnevaluatedConst<I>) -> Self;

    fn new_expr(interner: I, expr: I::ExprConst) -> Self;

    fn is_ct_var(self) -> bool {
        matches!(self.kind(), ty::ConstKind::Infer(ty::InferConst::Var(_)))
    }
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
}

pub trait Term<I: Interner<Term = Self>>:
    Copy + Debug + Hash + Eq + IntoKind<Kind = ty::TermKind<I>> + TypeFoldable<I> + Relate<I>
{
    fn as_type(&self) -> Option<I::Ty> {
        if let ty::TermKind::Ty(ty) = self.kind() { Some(ty) } else { None }
    }

    fn expect_type(&self) -> I::Ty {
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
}

pub trait GenericArgs<I: Interner<GenericArgs = Self>>:
    Copy
    + Debug
    + Hash
    + Eq
    + IntoIterator<Item = I::GenericArg>
    + Deref<Target: Deref<Target = [I::GenericArg]>>
    + Default
    + Relate<I>
{
    fn type_at(self, i: usize) -> I::Ty;

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
    + IntoKind<Kind = ty::Binder<I, ty::PredicateKind<I>>>
{
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
    // FIXME: Remove these, uplift the `Upcast` impls.
    + UpcastFrom<I, ty::Binder<I, ty::TraitRef<I>>>
    + UpcastFrom<I, ty::Binder<I, ty::ProjectionPredicate<I>>>
{
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

    fn is_phantom_data(self) -> bool;

    // FIXME: perhaps use `all_fields` and expose `FieldDef`.
    fn all_field_tys(self, interner: I) -> ty::EarlyBinder<I, impl Iterator<Item = I::Ty>>;

    fn sized_constraint(self, interner: I) -> Option<ty::EarlyBinder<I, I::Ty>>;
}

pub trait Features<I: Interner>: Copy {
    fn generic_const_exprs(self) -> bool;

    fn coroutine_clone(self) -> bool;
}
