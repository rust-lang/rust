//! This module contains `TyKind` and its major components.

#![allow(rustc::usage_of_ty_tykind)]

use std::assert_matches::debug_assert_matches;
use std::borrow::Cow;
use std::iter;
use std::ops::{ControlFlow, Range};

use hir::def::{CtorKind, DefKind};
use rustc_abi::{FIRST_VARIANT, FieldIdx, VariantIdx};
use rustc_errors::{ErrorGuaranteed, MultiSpan};
use rustc_hir as hir;
use rustc_hir::LangItem;
use rustc_hir::def_id::DefId;
use rustc_macros::{HashStable, TyDecodable, TyEncodable, TypeFoldable, extension};
use rustc_span::{DUMMY_SP, Span, Symbol, sym};
use rustc_type_ir::TyKind::*;
use rustc_type_ir::{self as ir, BoundVar, CollectAndApply, DynKind, TypeVisitableExt, elaborate};
use tracing::instrument;
use ty::util::{AsyncDropGlueMorphology, IntTypeExt};

use super::GenericParamDefKind;
use crate::infer::canonical::Canonical;
use crate::ty::InferTy::*;
use crate::ty::{
    self, AdtDef, BoundRegionKind, Discr, GenericArg, GenericArgs, GenericArgsRef, List, ParamEnv,
    Region, Ty, TyCtxt, TypeFlags, TypeSuperVisitable, TypeVisitable, TypeVisitor, UintTy,
};

// Re-export and re-parameterize some `I = TyCtxt<'tcx>` types here
#[rustc_diagnostic_item = "TyKind"]
pub type TyKind<'tcx> = ir::TyKind<TyCtxt<'tcx>>;
pub type TypeAndMut<'tcx> = ir::TypeAndMut<TyCtxt<'tcx>>;
pub type AliasTy<'tcx> = ir::AliasTy<TyCtxt<'tcx>>;
pub type FnSig<'tcx> = ir::FnSig<TyCtxt<'tcx>>;
pub type Binder<'tcx, T> = ir::Binder<TyCtxt<'tcx>, T>;
pub type EarlyBinder<'tcx, T> = ir::EarlyBinder<TyCtxt<'tcx>, T>;
pub type TypingMode<'tcx> = ir::TypingMode<TyCtxt<'tcx>>;

pub trait Article {
    fn article(&self) -> &'static str;
}

impl<'tcx> Article for TyKind<'tcx> {
    /// Get the article ("a" or "an") to use with this type.
    fn article(&self) -> &'static str {
        match self {
            Int(_) | Float(_) | Array(_, _) => "an",
            Adt(def, _) if def.is_enum() => "an",
            // This should never happen, but ICEing and causing the user's code
            // to not compile felt too harsh.
            Error(_) => "a",
            _ => "a",
        }
    }
}

#[extension(pub trait CoroutineArgsExt<'tcx>)]
impl<'tcx> ty::CoroutineArgs<TyCtxt<'tcx>> {
    /// Coroutine has not been resumed yet.
    const UNRESUMED: usize = 0;
    /// Coroutine has returned or is completed.
    const RETURNED: usize = 1;
    /// Coroutine has been poisoned.
    const POISONED: usize = 2;
    /// Number of variants to reserve in coroutine state. Corresponds to
    /// `UNRESUMED` (beginning of a coroutine) and `RETURNED`/`POISONED`
    /// (end of a coroutine) states.
    const RESERVED_VARIANTS: usize = 3;

    const UNRESUMED_NAME: &'static str = "Unresumed";
    const RETURNED_NAME: &'static str = "Returned";
    const POISONED_NAME: &'static str = "Panicked";

    /// The valid variant indices of this coroutine.
    #[inline]
    fn variant_range(&self, def_id: DefId, tcx: TyCtxt<'tcx>) -> Range<VariantIdx> {
        // FIXME requires optimized MIR
        FIRST_VARIANT
            ..tcx.coroutine_layout(def_id, tcx.types.unit).unwrap().variant_fields.next_index()
    }

    /// The discriminant for the given variant. Panics if the `variant_index` is
    /// out of range.
    #[inline]
    fn discriminant_for_variant(
        &self,
        def_id: DefId,
        tcx: TyCtxt<'tcx>,
        variant_index: VariantIdx,
    ) -> Discr<'tcx> {
        // Coroutines don't support explicit discriminant values, so they are
        // the same as the variant index.
        assert!(self.variant_range(def_id, tcx).contains(&variant_index));
        Discr { val: variant_index.as_usize() as u128, ty: self.discr_ty(tcx) }
    }

    /// The set of all discriminants for the coroutine, enumerated with their
    /// variant indices.
    #[inline]
    fn discriminants(
        self,
        def_id: DefId,
        tcx: TyCtxt<'tcx>,
    ) -> impl Iterator<Item = (VariantIdx, Discr<'tcx>)> {
        self.variant_range(def_id, tcx).map(move |index| {
            (index, Discr { val: index.as_usize() as u128, ty: self.discr_ty(tcx) })
        })
    }

    /// Calls `f` with a reference to the name of the enumerator for the given
    /// variant `v`.
    fn variant_name(v: VariantIdx) -> Cow<'static, str> {
        match v.as_usize() {
            Self::UNRESUMED => Cow::from(Self::UNRESUMED_NAME),
            Self::RETURNED => Cow::from(Self::RETURNED_NAME),
            Self::POISONED => Cow::from(Self::POISONED_NAME),
            _ => Cow::from(format!("Suspend{}", v.as_usize() - Self::RESERVED_VARIANTS)),
        }
    }

    /// The type of the state discriminant used in the coroutine type.
    #[inline]
    fn discr_ty(&self, tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        tcx.types.u32
    }

    /// This returns the types of the MIR locals which had to be stored across suspension points.
    /// It is calculated in rustc_mir_transform::coroutine::StateTransform.
    /// All the types here must be in the tuple in CoroutineInterior.
    ///
    /// The locals are grouped by their variant number. Note that some locals may
    /// be repeated in multiple variants.
    #[inline]
    fn state_tys(
        self,
        def_id: DefId,
        tcx: TyCtxt<'tcx>,
    ) -> impl Iterator<Item: Iterator<Item = Ty<'tcx>>> {
        let layout = tcx.coroutine_layout(def_id, self.kind_ty()).unwrap();
        layout.variant_fields.iter().map(move |variant| {
            variant.iter().map(move |field| {
                ty::EarlyBinder::bind(layout.field_tys[*field].ty).instantiate(tcx, self.args)
            })
        })
    }

    /// This is the types of the fields of a coroutine which are not stored in a
    /// variant.
    #[inline]
    fn prefix_tys(self) -> &'tcx List<Ty<'tcx>> {
        self.upvar_tys()
    }
}

#[derive(Debug, Copy, Clone, HashStable, TypeFoldable, TypeVisitable)]
pub enum UpvarArgs<'tcx> {
    Closure(GenericArgsRef<'tcx>),
    Coroutine(GenericArgsRef<'tcx>),
    CoroutineClosure(GenericArgsRef<'tcx>),
}

impl<'tcx> UpvarArgs<'tcx> {
    /// Returns an iterator over the list of types of captured paths by the closure/coroutine.
    /// In case there was a type error in figuring out the types of the captured path, an
    /// empty iterator is returned.
    #[inline]
    pub fn upvar_tys(self) -> &'tcx List<Ty<'tcx>> {
        let tupled_tys = match self {
            UpvarArgs::Closure(args) => args.as_closure().tupled_upvars_ty(),
            UpvarArgs::Coroutine(args) => args.as_coroutine().tupled_upvars_ty(),
            UpvarArgs::CoroutineClosure(args) => args.as_coroutine_closure().tupled_upvars_ty(),
        };

        match tupled_tys.kind() {
            TyKind::Error(_) => ty::List::empty(),
            TyKind::Tuple(..) => self.tupled_upvars_ty().tuple_fields(),
            TyKind::Infer(_) => bug!("upvar_tys called before capture types are inferred"),
            ty => bug!("Unexpected representation of upvar types tuple {:?}", ty),
        }
    }

    #[inline]
    pub fn tupled_upvars_ty(self) -> Ty<'tcx> {
        match self {
            UpvarArgs::Closure(args) => args.as_closure().tupled_upvars_ty(),
            UpvarArgs::Coroutine(args) => args.as_coroutine().tupled_upvars_ty(),
            UpvarArgs::CoroutineClosure(args) => args.as_coroutine_closure().tupled_upvars_ty(),
        }
    }
}

/// An inline const is modeled like
/// ```ignore (illustrative)
/// const InlineConst<'l0...'li, T0...Tj, R>: R;
/// ```
/// where:
///
/// - 'l0...'li and T0...Tj are the generic parameters
///   inherited from the item that defined the inline const,
/// - R represents the type of the constant.
///
/// When the inline const is instantiated, `R` is instantiated as the actual inferred
/// type of the constant. The reason that `R` is represented as an extra type parameter
/// is the same reason that [`ty::ClosureArgs`] have `CS` and `U` as type parameters:
/// inline const can reference lifetimes that are internal to the creating function.
#[derive(Copy, Clone, Debug)]
pub struct InlineConstArgs<'tcx> {
    /// Generic parameters from the enclosing item,
    /// concatenated with the inferred type of the constant.
    pub args: GenericArgsRef<'tcx>,
}

/// Struct returned by `split()`.
pub struct InlineConstArgsParts<'tcx, T> {
    pub parent_args: &'tcx [GenericArg<'tcx>],
    pub ty: T,
}

impl<'tcx> InlineConstArgs<'tcx> {
    /// Construct `InlineConstArgs` from `InlineConstArgsParts`.
    pub fn new(
        tcx: TyCtxt<'tcx>,
        parts: InlineConstArgsParts<'tcx, Ty<'tcx>>,
    ) -> InlineConstArgs<'tcx> {
        InlineConstArgs {
            args: tcx.mk_args_from_iter(
                parts.parent_args.iter().copied().chain(std::iter::once(parts.ty.into())),
            ),
        }
    }

    /// Divides the inline const args into their respective components.
    /// The ordering assumed here must match that used by `InlineConstArgs::new` above.
    fn split(self) -> InlineConstArgsParts<'tcx, GenericArg<'tcx>> {
        match self.args[..] {
            [ref parent_args @ .., ty] => InlineConstArgsParts { parent_args, ty },
            _ => bug!("inline const args missing synthetics"),
        }
    }

    /// Returns the generic parameters of the inline const's parent.
    pub fn parent_args(self) -> &'tcx [GenericArg<'tcx>] {
        self.split().parent_args
    }

    /// Returns the type of this inline const.
    pub fn ty(self) -> Ty<'tcx> {
        self.split().ty.expect_ty()
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, TyEncodable, TyDecodable)]
#[derive(HashStable)]
pub enum BoundVariableKind {
    Ty(BoundTyKind),
    Region(BoundRegionKind),
    Const,
}

impl BoundVariableKind {
    pub fn expect_region(self) -> BoundRegionKind {
        match self {
            BoundVariableKind::Region(lt) => lt,
            _ => bug!("expected a region, but found another kind"),
        }
    }

    pub fn expect_ty(self) -> BoundTyKind {
        match self {
            BoundVariableKind::Ty(ty) => ty,
            _ => bug!("expected a type, but found another kind"),
        }
    }

    pub fn expect_const(self) {
        match self {
            BoundVariableKind::Const => (),
            _ => bug!("expected a const, but found another kind"),
        }
    }
}

pub type PolyFnSig<'tcx> = Binder<'tcx, FnSig<'tcx>>;
pub type CanonicalPolyFnSig<'tcx> = Canonical<'tcx, Binder<'tcx, FnSig<'tcx>>>;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, TyEncodable, TyDecodable)]
#[derive(HashStable)]
pub struct ParamTy {
    pub index: u32,
    pub name: Symbol,
}

impl rustc_type_ir::inherent::ParamLike for ParamTy {
    fn index(self) -> u32 {
        self.index
    }
}

impl<'tcx> ParamTy {
    pub fn new(index: u32, name: Symbol) -> ParamTy {
        ParamTy { index, name }
    }

    pub fn for_def(def: &ty::GenericParamDef) -> ParamTy {
        ParamTy::new(def.index, def.name)
    }

    #[inline]
    pub fn to_ty(self, tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        Ty::new_param(tcx, self.index, self.name)
    }

    pub fn span_from_generics(self, tcx: TyCtxt<'tcx>, item_with_generics: DefId) -> Span {
        let generics = tcx.generics_of(item_with_generics);
        let type_param = generics.type_param(self, tcx);
        tcx.def_span(type_param.def_id)
    }
}

#[derive(Copy, Clone, Hash, TyEncodable, TyDecodable, Eq, PartialEq, Ord, PartialOrd)]
#[derive(HashStable)]
pub struct ParamConst {
    pub index: u32,
    pub name: Symbol,
}

impl rustc_type_ir::inherent::ParamLike for ParamConst {
    fn index(self) -> u32 {
        self.index
    }
}

impl ParamConst {
    pub fn new(index: u32, name: Symbol) -> ParamConst {
        ParamConst { index, name }
    }

    pub fn for_def(def: &ty::GenericParamDef) -> ParamConst {
        ParamConst::new(def.index, def.name)
    }

    #[instrument(level = "debug")]
    pub fn find_ty_from_env<'tcx>(self, env: ParamEnv<'tcx>) -> Ty<'tcx> {
        let mut candidates = env.caller_bounds().iter().filter_map(|clause| {
            // `ConstArgHasType` are never desugared to be higher ranked.
            match clause.kind().skip_binder() {
                ty::ClauseKind::ConstArgHasType(param_ct, ty) => {
                    assert!(!(param_ct, ty).has_escaping_bound_vars());

                    match param_ct.kind() {
                        ty::ConstKind::Param(param_ct) if param_ct.index == self.index => Some(ty),
                        _ => None,
                    }
                }
                _ => None,
            }
        });

        let ty = candidates.next().unwrap();
        assert!(candidates.next().is_none());
        ty
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, TyEncodable, TyDecodable)]
#[derive(HashStable)]
pub struct BoundTy {
    pub var: BoundVar,
    pub kind: BoundTyKind,
}

impl<'tcx> rustc_type_ir::inherent::BoundVarLike<TyCtxt<'tcx>> for BoundTy {
    fn var(self) -> BoundVar {
        self.var
    }

    fn assert_eq(self, var: ty::BoundVariableKind) {
        assert_eq!(self.kind, var.expect_ty())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, TyEncodable, TyDecodable)]
#[derive(HashStable)]
pub enum BoundTyKind {
    Anon,
    Param(DefId, Symbol),
}

impl From<BoundVar> for BoundTy {
    fn from(var: BoundVar) -> Self {
        BoundTy { var, kind: BoundTyKind::Anon }
    }
}

/// Constructors for `Ty`
impl<'tcx> Ty<'tcx> {
    /// Avoid using this in favour of more specific `new_*` methods, where possible.
    /// The more specific methods will often optimize their creation.
    #[allow(rustc::usage_of_ty_tykind)]
    #[inline]
    fn new(tcx: TyCtxt<'tcx>, st: TyKind<'tcx>) -> Ty<'tcx> {
        tcx.mk_ty_from_kind(st)
    }

    #[inline]
    pub fn new_infer(tcx: TyCtxt<'tcx>, infer: ty::InferTy) -> Ty<'tcx> {
        Ty::new(tcx, TyKind::Infer(infer))
    }

    #[inline]
    pub fn new_var(tcx: TyCtxt<'tcx>, v: ty::TyVid) -> Ty<'tcx> {
        // Use a pre-interned one when possible.
        tcx.types
            .ty_vars
            .get(v.as_usize())
            .copied()
            .unwrap_or_else(|| Ty::new(tcx, Infer(TyVar(v))))
    }

    #[inline]
    pub fn new_int_var(tcx: TyCtxt<'tcx>, v: ty::IntVid) -> Ty<'tcx> {
        Ty::new_infer(tcx, IntVar(v))
    }

    #[inline]
    pub fn new_float_var(tcx: TyCtxt<'tcx>, v: ty::FloatVid) -> Ty<'tcx> {
        Ty::new_infer(tcx, FloatVar(v))
    }

    #[inline]
    pub fn new_fresh(tcx: TyCtxt<'tcx>, n: u32) -> Ty<'tcx> {
        // Use a pre-interned one when possible.
        tcx.types
            .fresh_tys
            .get(n as usize)
            .copied()
            .unwrap_or_else(|| Ty::new_infer(tcx, ty::FreshTy(n)))
    }

    #[inline]
    pub fn new_fresh_int(tcx: TyCtxt<'tcx>, n: u32) -> Ty<'tcx> {
        // Use a pre-interned one when possible.
        tcx.types
            .fresh_int_tys
            .get(n as usize)
            .copied()
            .unwrap_or_else(|| Ty::new_infer(tcx, ty::FreshIntTy(n)))
    }

    #[inline]
    pub fn new_fresh_float(tcx: TyCtxt<'tcx>, n: u32) -> Ty<'tcx> {
        // Use a pre-interned one when possible.
        tcx.types
            .fresh_float_tys
            .get(n as usize)
            .copied()
            .unwrap_or_else(|| Ty::new_infer(tcx, ty::FreshFloatTy(n)))
    }

    #[inline]
    pub fn new_param(tcx: TyCtxt<'tcx>, index: u32, name: Symbol) -> Ty<'tcx> {
        Ty::new(tcx, Param(ParamTy { index, name }))
    }

    #[inline]
    pub fn new_bound(
        tcx: TyCtxt<'tcx>,
        index: ty::DebruijnIndex,
        bound_ty: ty::BoundTy,
    ) -> Ty<'tcx> {
        Ty::new(tcx, Bound(index, bound_ty))
    }

    #[inline]
    pub fn new_placeholder(tcx: TyCtxt<'tcx>, placeholder: ty::PlaceholderType) -> Ty<'tcx> {
        Ty::new(tcx, Placeholder(placeholder))
    }

    #[inline]
    pub fn new_alias(
        tcx: TyCtxt<'tcx>,
        kind: ty::AliasTyKind,
        alias_ty: ty::AliasTy<'tcx>,
    ) -> Ty<'tcx> {
        debug_assert_matches!(
            (kind, tcx.def_kind(alias_ty.def_id)),
            (ty::Opaque, DefKind::OpaqueTy)
                | (ty::Projection | ty::Inherent, DefKind::AssocTy)
                | (ty::Weak, DefKind::TyAlias)
        );
        Ty::new(tcx, Alias(kind, alias_ty))
    }

    #[inline]
    pub fn new_pat(tcx: TyCtxt<'tcx>, base: Ty<'tcx>, pat: ty::Pattern<'tcx>) -> Ty<'tcx> {
        Ty::new(tcx, Pat(base, pat))
    }

    #[inline]
    #[instrument(level = "debug", skip(tcx))]
    pub fn new_opaque(tcx: TyCtxt<'tcx>, def_id: DefId, args: GenericArgsRef<'tcx>) -> Ty<'tcx> {
        Ty::new_alias(tcx, ty::Opaque, AliasTy::new_from_args(tcx, def_id, args))
    }

    /// Constructs a `TyKind::Error` type with current `ErrorGuaranteed`
    pub fn new_error(tcx: TyCtxt<'tcx>, guar: ErrorGuaranteed) -> Ty<'tcx> {
        Ty::new(tcx, Error(guar))
    }

    /// Constructs a `TyKind::Error` type and registers a `span_delayed_bug` to ensure it gets used.
    #[track_caller]
    pub fn new_misc_error(tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        Ty::new_error_with_message(tcx, DUMMY_SP, "TyKind::Error constructed but no error reported")
    }

    /// Constructs a `TyKind::Error` type and registers a `span_delayed_bug` with the given `msg` to
    /// ensure it gets used.
    #[track_caller]
    pub fn new_error_with_message<S: Into<MultiSpan>>(
        tcx: TyCtxt<'tcx>,
        span: S,
        msg: impl Into<Cow<'static, str>>,
    ) -> Ty<'tcx> {
        let reported = tcx.dcx().span_delayed_bug(span, msg);
        Ty::new(tcx, Error(reported))
    }

    #[inline]
    pub fn new_int(tcx: TyCtxt<'tcx>, i: ty::IntTy) -> Ty<'tcx> {
        use ty::IntTy::*;
        match i {
            Isize => tcx.types.isize,
            I8 => tcx.types.i8,
            I16 => tcx.types.i16,
            I32 => tcx.types.i32,
            I64 => tcx.types.i64,
            I128 => tcx.types.i128,
        }
    }

    #[inline]
    pub fn new_uint(tcx: TyCtxt<'tcx>, ui: ty::UintTy) -> Ty<'tcx> {
        use ty::UintTy::*;
        match ui {
            Usize => tcx.types.usize,
            U8 => tcx.types.u8,
            U16 => tcx.types.u16,
            U32 => tcx.types.u32,
            U64 => tcx.types.u64,
            U128 => tcx.types.u128,
        }
    }

    #[inline]
    pub fn new_float(tcx: TyCtxt<'tcx>, f: ty::FloatTy) -> Ty<'tcx> {
        use ty::FloatTy::*;
        match f {
            F16 => tcx.types.f16,
            F32 => tcx.types.f32,
            F64 => tcx.types.f64,
            F128 => tcx.types.f128,
        }
    }

    #[inline]
    pub fn new_ref(
        tcx: TyCtxt<'tcx>,
        r: Region<'tcx>,
        ty: Ty<'tcx>,
        mutbl: ty::Mutability,
    ) -> Ty<'tcx> {
        Ty::new(tcx, Ref(r, ty, mutbl))
    }

    #[inline]
    pub fn new_mut_ref(tcx: TyCtxt<'tcx>, r: Region<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
        Ty::new_ref(tcx, r, ty, hir::Mutability::Mut)
    }

    #[inline]
    pub fn new_imm_ref(tcx: TyCtxt<'tcx>, r: Region<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
        Ty::new_ref(tcx, r, ty, hir::Mutability::Not)
    }

    pub fn new_pinned_ref(
        tcx: TyCtxt<'tcx>,
        r: Region<'tcx>,
        ty: Ty<'tcx>,
        mutbl: ty::Mutability,
    ) -> Ty<'tcx> {
        let pin = tcx.adt_def(tcx.require_lang_item(LangItem::Pin, None));
        Ty::new_adt(tcx, pin, tcx.mk_args(&[Ty::new_ref(tcx, r, ty, mutbl).into()]))
    }

    #[inline]
    pub fn new_ptr(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>, mutbl: ty::Mutability) -> Ty<'tcx> {
        Ty::new(tcx, ty::RawPtr(ty, mutbl))
    }

    #[inline]
    pub fn new_mut_ptr(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
        Ty::new_ptr(tcx, ty, hir::Mutability::Mut)
    }

    #[inline]
    pub fn new_imm_ptr(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
        Ty::new_ptr(tcx, ty, hir::Mutability::Not)
    }

    #[inline]
    pub fn new_adt(tcx: TyCtxt<'tcx>, def: AdtDef<'tcx>, args: GenericArgsRef<'tcx>) -> Ty<'tcx> {
        tcx.debug_assert_args_compatible(def.did(), args);
        if cfg!(debug_assertions) {
            match tcx.def_kind(def.did()) {
                DefKind::Struct | DefKind::Union | DefKind::Enum => {}
                DefKind::Mod
                | DefKind::Variant
                | DefKind::Trait
                | DefKind::TyAlias
                | DefKind::ForeignTy
                | DefKind::TraitAlias
                | DefKind::AssocTy
                | DefKind::TyParam
                | DefKind::Fn
                | DefKind::Const
                | DefKind::ConstParam
                | DefKind::Static { .. }
                | DefKind::Ctor(..)
                | DefKind::AssocFn
                | DefKind::AssocConst
                | DefKind::Macro(..)
                | DefKind::ExternCrate
                | DefKind::Use
                | DefKind::ForeignMod
                | DefKind::AnonConst
                | DefKind::InlineConst
                | DefKind::OpaqueTy
                | DefKind::Field
                | DefKind::LifetimeParam
                | DefKind::GlobalAsm
                | DefKind::Impl { .. }
                | DefKind::Closure
                | DefKind::SyntheticCoroutineBody => {
                    bug!("not an adt: {def:?} ({:?})", tcx.def_kind(def.did()))
                }
            }
        }
        Ty::new(tcx, Adt(def, args))
    }

    #[inline]
    pub fn new_foreign(tcx: TyCtxt<'tcx>, def_id: DefId) -> Ty<'tcx> {
        Ty::new(tcx, Foreign(def_id))
    }

    #[inline]
    pub fn new_array(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>, n: u64) -> Ty<'tcx> {
        Ty::new(tcx, Array(ty, ty::Const::from_target_usize(tcx, n)))
    }

    #[inline]
    pub fn new_array_with_const_len(
        tcx: TyCtxt<'tcx>,
        ty: Ty<'tcx>,
        ct: ty::Const<'tcx>,
    ) -> Ty<'tcx> {
        Ty::new(tcx, Array(ty, ct))
    }

    #[inline]
    pub fn new_slice(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
        Ty::new(tcx, Slice(ty))
    }

    #[inline]
    pub fn new_tup(tcx: TyCtxt<'tcx>, ts: &[Ty<'tcx>]) -> Ty<'tcx> {
        if ts.is_empty() { tcx.types.unit } else { Ty::new(tcx, Tuple(tcx.mk_type_list(ts))) }
    }

    pub fn new_tup_from_iter<I, T>(tcx: TyCtxt<'tcx>, iter: I) -> T::Output
    where
        I: Iterator<Item = T>,
        T: CollectAndApply<Ty<'tcx>, Ty<'tcx>>,
    {
        T::collect_and_apply(iter, |ts| Ty::new_tup(tcx, ts))
    }

    #[inline]
    pub fn new_fn_def(
        tcx: TyCtxt<'tcx>,
        def_id: DefId,
        args: impl IntoIterator<Item: Into<GenericArg<'tcx>>>,
    ) -> Ty<'tcx> {
        debug_assert_matches!(
            tcx.def_kind(def_id),
            DefKind::AssocFn | DefKind::Fn | DefKind::Ctor(_, CtorKind::Fn)
        );
        let args = tcx.check_and_mk_args(def_id, args);
        Ty::new(tcx, FnDef(def_id, args))
    }

    #[inline]
    pub fn new_fn_ptr(tcx: TyCtxt<'tcx>, fty: PolyFnSig<'tcx>) -> Ty<'tcx> {
        let (sig_tys, hdr) = fty.split();
        Ty::new(tcx, FnPtr(sig_tys, hdr))
    }

    #[inline]
    pub fn new_unsafe_binder(tcx: TyCtxt<'tcx>, b: Binder<'tcx, Ty<'tcx>>) -> Ty<'tcx> {
        Ty::new(tcx, UnsafeBinder(b.into()))
    }

    #[inline]
    pub fn new_dynamic(
        tcx: TyCtxt<'tcx>,
        obj: &'tcx List<ty::PolyExistentialPredicate<'tcx>>,
        reg: ty::Region<'tcx>,
        repr: DynKind,
    ) -> Ty<'tcx> {
        if cfg!(debug_assertions) {
            let projection_count = obj.projection_bounds().count();
            let expected_count: usize = obj
                .principal_def_id()
                .into_iter()
                .flat_map(|principal_def_id| {
                    // NOTE: This should agree with `needed_associated_types` in
                    // dyn trait lowering, or else we'll have ICEs.
                    elaborate::supertraits(
                        tcx,
                        ty::Binder::dummy(ty::TraitRef::identity(tcx, principal_def_id)),
                    )
                    .map(|principal| {
                        tcx.associated_items(principal.def_id())
                            .in_definition_order()
                            .filter(|item| item.kind == ty::AssocKind::Type)
                            .filter(|item| !item.is_impl_trait_in_trait())
                            .filter(|item| !tcx.generics_require_sized_self(item.def_id))
                            .count()
                    })
                })
                .sum();
            assert_eq!(
                projection_count, expected_count,
                "expected {obj:?} to have {expected_count} projections, \
                but it has {projection_count}"
            );
        }
        Ty::new(tcx, Dynamic(obj, reg, repr))
    }

    #[inline]
    pub fn new_projection_from_args(
        tcx: TyCtxt<'tcx>,
        item_def_id: DefId,
        args: ty::GenericArgsRef<'tcx>,
    ) -> Ty<'tcx> {
        Ty::new_alias(tcx, ty::Projection, AliasTy::new_from_args(tcx, item_def_id, args))
    }

    #[inline]
    pub fn new_projection(
        tcx: TyCtxt<'tcx>,
        item_def_id: DefId,
        args: impl IntoIterator<Item: Into<GenericArg<'tcx>>>,
    ) -> Ty<'tcx> {
        Ty::new_alias(tcx, ty::Projection, AliasTy::new(tcx, item_def_id, args))
    }

    #[inline]
    pub fn new_closure(
        tcx: TyCtxt<'tcx>,
        def_id: DefId,
        closure_args: GenericArgsRef<'tcx>,
    ) -> Ty<'tcx> {
        tcx.debug_assert_args_compatible(def_id, closure_args);
        Ty::new(tcx, Closure(def_id, closure_args))
    }

    #[inline]
    pub fn new_coroutine_closure(
        tcx: TyCtxt<'tcx>,
        def_id: DefId,
        closure_args: GenericArgsRef<'tcx>,
    ) -> Ty<'tcx> {
        tcx.debug_assert_args_compatible(def_id, closure_args);
        Ty::new(tcx, CoroutineClosure(def_id, closure_args))
    }

    #[inline]
    pub fn new_coroutine(
        tcx: TyCtxt<'tcx>,
        def_id: DefId,
        coroutine_args: GenericArgsRef<'tcx>,
    ) -> Ty<'tcx> {
        tcx.debug_assert_args_compatible(def_id, coroutine_args);
        Ty::new(tcx, Coroutine(def_id, coroutine_args))
    }

    #[inline]
    pub fn new_coroutine_witness(
        tcx: TyCtxt<'tcx>,
        id: DefId,
        args: GenericArgsRef<'tcx>,
    ) -> Ty<'tcx> {
        Ty::new(tcx, CoroutineWitness(id, args))
    }

    // misc

    #[inline]
    pub fn new_static_str(tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        Ty::new_imm_ref(tcx, tcx.lifetimes.re_static, tcx.types.str_)
    }

    #[inline]
    pub fn new_diverging_default(tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        if tcx.features().never_type_fallback() { tcx.types.never } else { tcx.types.unit }
    }

    // lang and diagnostic tys

    fn new_generic_adt(tcx: TyCtxt<'tcx>, wrapper_def_id: DefId, ty_param: Ty<'tcx>) -> Ty<'tcx> {
        let adt_def = tcx.adt_def(wrapper_def_id);
        let args = GenericArgs::for_item(tcx, wrapper_def_id, |param, args| match param.kind {
            GenericParamDefKind::Lifetime | GenericParamDefKind::Const { .. } => bug!(),
            GenericParamDefKind::Type { has_default, .. } => {
                if param.index == 0 {
                    ty_param.into()
                } else {
                    assert!(has_default);
                    tcx.type_of(param.def_id).instantiate(tcx, args).into()
                }
            }
        });
        Ty::new_adt(tcx, adt_def, args)
    }

    #[inline]
    pub fn new_lang_item(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>, item: LangItem) -> Option<Ty<'tcx>> {
        let def_id = tcx.lang_items().get(item)?;
        Some(Ty::new_generic_adt(tcx, def_id, ty))
    }

    #[inline]
    pub fn new_diagnostic_item(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>, name: Symbol) -> Option<Ty<'tcx>> {
        let def_id = tcx.get_diagnostic_item(name)?;
        Some(Ty::new_generic_adt(tcx, def_id, ty))
    }

    #[inline]
    pub fn new_box(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
        let def_id = tcx.require_lang_item(LangItem::OwnedBox, None);
        Ty::new_generic_adt(tcx, def_id, ty)
    }

    #[inline]
    pub fn new_maybe_uninit(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
        let def_id = tcx.require_lang_item(LangItem::MaybeUninit, None);
        Ty::new_generic_adt(tcx, def_id, ty)
    }

    /// Creates a `&mut Context<'_>` [`Ty`] with erased lifetimes.
    pub fn new_task_context(tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        let context_did = tcx.require_lang_item(LangItem::Context, None);
        let context_adt_ref = tcx.adt_def(context_did);
        let context_args = tcx.mk_args(&[tcx.lifetimes.re_erased.into()]);
        let context_ty = Ty::new_adt(tcx, context_adt_ref, context_args);
        Ty::new_mut_ref(tcx, tcx.lifetimes.re_erased, context_ty)
    }
}

impl<'tcx> rustc_type_ir::inherent::Ty<TyCtxt<'tcx>> for Ty<'tcx> {
    fn new_bool(tcx: TyCtxt<'tcx>) -> Self {
        tcx.types.bool
    }

    fn new_u8(tcx: TyCtxt<'tcx>) -> Self {
        tcx.types.u8
    }

    fn new_infer(tcx: TyCtxt<'tcx>, infer: ty::InferTy) -> Self {
        Ty::new_infer(tcx, infer)
    }

    fn new_var(tcx: TyCtxt<'tcx>, vid: ty::TyVid) -> Self {
        Ty::new_var(tcx, vid)
    }

    fn new_param(tcx: TyCtxt<'tcx>, param: ty::ParamTy) -> Self {
        Ty::new_param(tcx, param.index, param.name)
    }

    fn new_placeholder(tcx: TyCtxt<'tcx>, placeholder: ty::PlaceholderType) -> Self {
        Ty::new_placeholder(tcx, placeholder)
    }

    fn new_bound(interner: TyCtxt<'tcx>, debruijn: ty::DebruijnIndex, var: ty::BoundTy) -> Self {
        Ty::new_bound(interner, debruijn, var)
    }

    fn new_anon_bound(tcx: TyCtxt<'tcx>, debruijn: ty::DebruijnIndex, var: ty::BoundVar) -> Self {
        Ty::new_bound(tcx, debruijn, ty::BoundTy { var, kind: ty::BoundTyKind::Anon })
    }

    fn new_alias(
        interner: TyCtxt<'tcx>,
        kind: ty::AliasTyKind,
        alias_ty: ty::AliasTy<'tcx>,
    ) -> Self {
        Ty::new_alias(interner, kind, alias_ty)
    }

    fn new_error(interner: TyCtxt<'tcx>, guar: ErrorGuaranteed) -> Self {
        Ty::new_error(interner, guar)
    }

    fn new_adt(
        interner: TyCtxt<'tcx>,
        adt_def: ty::AdtDef<'tcx>,
        args: ty::GenericArgsRef<'tcx>,
    ) -> Self {
        Ty::new_adt(interner, adt_def, args)
    }

    fn new_foreign(interner: TyCtxt<'tcx>, def_id: DefId) -> Self {
        Ty::new_foreign(interner, def_id)
    }

    fn new_dynamic(
        interner: TyCtxt<'tcx>,
        preds: &'tcx List<ty::PolyExistentialPredicate<'tcx>>,
        region: ty::Region<'tcx>,
        kind: ty::DynKind,
    ) -> Self {
        Ty::new_dynamic(interner, preds, region, kind)
    }

    fn new_coroutine(
        interner: TyCtxt<'tcx>,
        def_id: DefId,
        args: ty::GenericArgsRef<'tcx>,
    ) -> Self {
        Ty::new_coroutine(interner, def_id, args)
    }

    fn new_coroutine_closure(
        interner: TyCtxt<'tcx>,
        def_id: DefId,
        args: ty::GenericArgsRef<'tcx>,
    ) -> Self {
        Ty::new_coroutine_closure(interner, def_id, args)
    }

    fn new_closure(interner: TyCtxt<'tcx>, def_id: DefId, args: ty::GenericArgsRef<'tcx>) -> Self {
        Ty::new_closure(interner, def_id, args)
    }

    fn new_coroutine_witness(
        interner: TyCtxt<'tcx>,
        def_id: DefId,
        args: ty::GenericArgsRef<'tcx>,
    ) -> Self {
        Ty::new_coroutine_witness(interner, def_id, args)
    }

    fn new_ptr(interner: TyCtxt<'tcx>, ty: Self, mutbl: hir::Mutability) -> Self {
        Ty::new_ptr(interner, ty, mutbl)
    }

    fn new_ref(
        interner: TyCtxt<'tcx>,
        region: ty::Region<'tcx>,
        ty: Self,
        mutbl: hir::Mutability,
    ) -> Self {
        Ty::new_ref(interner, region, ty, mutbl)
    }

    fn new_array_with_const_len(interner: TyCtxt<'tcx>, ty: Self, len: ty::Const<'tcx>) -> Self {
        Ty::new_array_with_const_len(interner, ty, len)
    }

    fn new_slice(interner: TyCtxt<'tcx>, ty: Self) -> Self {
        Ty::new_slice(interner, ty)
    }

    fn new_tup(interner: TyCtxt<'tcx>, tys: &[Ty<'tcx>]) -> Self {
        Ty::new_tup(interner, tys)
    }

    fn new_tup_from_iter<It, T>(interner: TyCtxt<'tcx>, iter: It) -> T::Output
    where
        It: Iterator<Item = T>,
        T: CollectAndApply<Self, Self>,
    {
        Ty::new_tup_from_iter(interner, iter)
    }

    fn tuple_fields(self) -> &'tcx ty::List<Ty<'tcx>> {
        self.tuple_fields()
    }

    fn to_opt_closure_kind(self) -> Option<ty::ClosureKind> {
        self.to_opt_closure_kind()
    }

    fn from_closure_kind(interner: TyCtxt<'tcx>, kind: ty::ClosureKind) -> Self {
        Ty::from_closure_kind(interner, kind)
    }

    fn from_coroutine_closure_kind(
        interner: TyCtxt<'tcx>,
        kind: rustc_type_ir::ClosureKind,
    ) -> Self {
        Ty::from_coroutine_closure_kind(interner, kind)
    }

    fn new_fn_def(interner: TyCtxt<'tcx>, def_id: DefId, args: ty::GenericArgsRef<'tcx>) -> Self {
        Ty::new_fn_def(interner, def_id, args)
    }

    fn new_fn_ptr(interner: TyCtxt<'tcx>, sig: ty::Binder<'tcx, ty::FnSig<'tcx>>) -> Self {
        Ty::new_fn_ptr(interner, sig)
    }

    fn new_pat(interner: TyCtxt<'tcx>, ty: Self, pat: ty::Pattern<'tcx>) -> Self {
        Ty::new_pat(interner, ty, pat)
    }

    fn new_unsafe_binder(interner: TyCtxt<'tcx>, ty: ty::Binder<'tcx, Ty<'tcx>>) -> Self {
        Ty::new_unsafe_binder(interner, ty)
    }

    fn new_unit(interner: TyCtxt<'tcx>) -> Self {
        interner.types.unit
    }

    fn new_usize(interner: TyCtxt<'tcx>) -> Self {
        interner.types.usize
    }

    fn discriminant_ty(self, interner: TyCtxt<'tcx>) -> Ty<'tcx> {
        self.discriminant_ty(interner)
    }

    fn async_destructor_ty(self, interner: TyCtxt<'tcx>) -> Ty<'tcx> {
        self.async_destructor_ty(interner)
    }

    fn has_unsafe_fields(self) -> bool {
        Ty::has_unsafe_fields(self)
    }
}

/// Type utilities
impl<'tcx> Ty<'tcx> {
    // It would be nicer if this returned the value instead of a reference,
    // like how `Predicate::kind` and `Region::kind` do. (It would result in
    // many fewer subsequent dereferences.) But that gives a small but
    // noticeable performance hit. See #126069 for details.
    #[inline(always)]
    pub fn kind(self) -> &'tcx TyKind<'tcx> {
        self.0.0
    }

    // FIXME(compiler-errors): Think about removing this.
    #[inline(always)]
    pub fn flags(self) -> TypeFlags {
        self.0.0.flags
    }

    #[inline]
    pub fn is_unit(self) -> bool {
        match self.kind() {
            Tuple(tys) => tys.is_empty(),
            _ => false,
        }
    }

    /// Check if type is an `usize`.
    #[inline]
    pub fn is_usize(self) -> bool {
        matches!(self.kind(), Uint(UintTy::Usize))
    }

    /// Check if type is an `usize` or an integral type variable.
    #[inline]
    pub fn is_usize_like(self) -> bool {
        matches!(self.kind(), Uint(UintTy::Usize) | Infer(IntVar(_)))
    }

    #[inline]
    pub fn is_never(self) -> bool {
        matches!(self.kind(), Never)
    }

    #[inline]
    pub fn is_primitive(self) -> bool {
        matches!(self.kind(), Bool | Char | Int(_) | Uint(_) | Float(_))
    }

    #[inline]
    pub fn is_adt(self) -> bool {
        matches!(self.kind(), Adt(..))
    }

    #[inline]
    pub fn is_ref(self) -> bool {
        matches!(self.kind(), Ref(..))
    }

    #[inline]
    pub fn is_ty_var(self) -> bool {
        matches!(self.kind(), Infer(TyVar(_)))
    }

    #[inline]
    pub fn ty_vid(self) -> Option<ty::TyVid> {
        match self.kind() {
            &Infer(TyVar(vid)) => Some(vid),
            _ => None,
        }
    }

    #[inline]
    pub fn is_ty_or_numeric_infer(self) -> bool {
        matches!(self.kind(), Infer(_))
    }

    #[inline]
    pub fn is_phantom_data(self) -> bool {
        if let Adt(def, _) = self.kind() { def.is_phantom_data() } else { false }
    }

    #[inline]
    pub fn is_bool(self) -> bool {
        *self.kind() == Bool
    }

    /// Returns `true` if this type is a `str`.
    #[inline]
    pub fn is_str(self) -> bool {
        *self.kind() == Str
    }

    #[inline]
    pub fn is_param(self, index: u32) -> bool {
        match self.kind() {
            ty::Param(data) => data.index == index,
            _ => false,
        }
    }

    #[inline]
    pub fn is_slice(self) -> bool {
        matches!(self.kind(), Slice(_))
    }

    #[inline]
    pub fn is_array_slice(self) -> bool {
        match self.kind() {
            Slice(_) => true,
            ty::RawPtr(ty, _) | Ref(_, ty, _) => matches!(ty.kind(), Slice(_)),
            _ => false,
        }
    }

    #[inline]
    pub fn is_array(self) -> bool {
        matches!(self.kind(), Array(..))
    }

    #[inline]
    pub fn is_simd(self) -> bool {
        match self.kind() {
            Adt(def, _) => def.repr().simd(),
            _ => false,
        }
    }

    pub fn sequence_element_type(self, tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        match self.kind() {
            Array(ty, _) | Slice(ty) => *ty,
            Str => tcx.types.u8,
            _ => bug!("`sequence_element_type` called on non-sequence value: {}", self),
        }
    }

    pub fn simd_size_and_type(self, tcx: TyCtxt<'tcx>) -> (u64, Ty<'tcx>) {
        let Adt(def, args) = self.kind() else {
            bug!("`simd_size_and_type` called on invalid type")
        };
        assert!(def.repr().simd(), "`simd_size_and_type` called on non-SIMD type");
        let variant = def.non_enum_variant();
        assert_eq!(variant.fields.len(), 1);
        let field_ty = variant.fields[FieldIdx::ZERO].ty(tcx, args);
        let Array(f0_elem_ty, f0_len) = field_ty.kind() else {
            bug!("Simd type has non-array field type {field_ty:?}")
        };
        // FIXME(repr_simd): https://github.com/rust-lang/rust/pull/78863#discussion_r522784112
        // The way we evaluate the `N` in `[T; N]` here only works since we use
        // `simd_size_and_type` post-monomorphization. It will probably start to ICE
        // if we use it in generic code. See the `simd-array-trait` ui test.
        (
            f0_len
                .try_to_target_usize(tcx)
                .expect("expected SIMD field to have definite array size"),
            *f0_elem_ty,
        )
    }

    #[inline]
    pub fn is_mutable_ptr(self) -> bool {
        matches!(self.kind(), RawPtr(_, hir::Mutability::Mut) | Ref(_, _, hir::Mutability::Mut))
    }

    /// Get the mutability of the reference or `None` when not a reference
    #[inline]
    pub fn ref_mutability(self) -> Option<hir::Mutability> {
        match self.kind() {
            Ref(_, _, mutability) => Some(*mutability),
            _ => None,
        }
    }

    #[inline]
    pub fn is_raw_ptr(self) -> bool {
        matches!(self.kind(), RawPtr(_, _))
    }

    /// Tests if this is any kind of primitive pointer type (reference, raw pointer, fn pointer).
    /// `Box` is *not* considered a pointer here!
    #[inline]
    pub fn is_any_ptr(self) -> bool {
        self.is_ref() || self.is_raw_ptr() || self.is_fn_ptr()
    }

    #[inline]
    pub fn is_box(self) -> bool {
        match self.kind() {
            Adt(def, _) => def.is_box(),
            _ => false,
        }
    }

    /// Tests whether this is a Box definitely using the global allocator.
    ///
    /// If the allocator is still generic, the answer is `false`, but it may
    /// later turn out that it does use the global allocator.
    #[inline]
    pub fn is_box_global(self, tcx: TyCtxt<'tcx>) -> bool {
        match self.kind() {
            Adt(def, args) if def.is_box() => {
                let Some(alloc) = args.get(1) else {
                    // Single-argument Box is always global. (for "minicore" tests)
                    return true;
                };
                alloc.expect_ty().ty_adt_def().is_some_and(|alloc_adt| {
                    let global_alloc = tcx.require_lang_item(LangItem::GlobalAlloc, None);
                    alloc_adt.did() == global_alloc
                })
            }
            _ => false,
        }
    }

    pub fn boxed_ty(self) -> Option<Ty<'tcx>> {
        match self.kind() {
            Adt(def, args) if def.is_box() => Some(args.type_at(0)),
            _ => None,
        }
    }

    /// Panics if called on any type other than `Box<T>`.
    pub fn expect_boxed_ty(self) -> Ty<'tcx> {
        self.boxed_ty()
            .unwrap_or_else(|| bug!("`expect_boxed_ty` is called on non-box type {:?}", self))
    }

    /// A scalar type is one that denotes an atomic datum, with no sub-components.
    /// (A RawPtr is scalar because it represents a non-managed pointer, so its
    /// contents are abstract to rustc.)
    #[inline]
    pub fn is_scalar(self) -> bool {
        matches!(
            self.kind(),
            Bool | Char
                | Int(_)
                | Float(_)
                | Uint(_)
                | FnDef(..)
                | FnPtr(..)
                | RawPtr(_, _)
                | Infer(IntVar(_) | FloatVar(_))
        )
    }

    /// Returns `true` if this type is a floating point type.
    #[inline]
    pub fn is_floating_point(self) -> bool {
        matches!(self.kind(), Float(_) | Infer(FloatVar(_)))
    }

    #[inline]
    pub fn is_trait(self) -> bool {
        matches!(self.kind(), Dynamic(_, _, ty::Dyn))
    }

    #[inline]
    pub fn is_dyn_star(self) -> bool {
        matches!(self.kind(), Dynamic(_, _, ty::DynStar))
    }

    #[inline]
    pub fn is_enum(self) -> bool {
        matches!(self.kind(), Adt(adt_def, _) if adt_def.is_enum())
    }

    #[inline]
    pub fn is_union(self) -> bool {
        matches!(self.kind(), Adt(adt_def, _) if adt_def.is_union())
    }

    #[inline]
    pub fn is_closure(self) -> bool {
        matches!(self.kind(), Closure(..))
    }

    #[inline]
    pub fn is_coroutine(self) -> bool {
        matches!(self.kind(), Coroutine(..))
    }

    #[inline]
    pub fn is_coroutine_closure(self) -> bool {
        matches!(self.kind(), CoroutineClosure(..))
    }

    #[inline]
    pub fn is_integral(self) -> bool {
        matches!(self.kind(), Infer(IntVar(_)) | Int(_) | Uint(_))
    }

    #[inline]
    pub fn is_fresh_ty(self) -> bool {
        matches!(self.kind(), Infer(FreshTy(_)))
    }

    #[inline]
    pub fn is_fresh(self) -> bool {
        matches!(self.kind(), Infer(FreshTy(_) | FreshIntTy(_) | FreshFloatTy(_)))
    }

    #[inline]
    pub fn is_char(self) -> bool {
        matches!(self.kind(), Char)
    }

    #[inline]
    pub fn is_numeric(self) -> bool {
        self.is_integral() || self.is_floating_point()
    }

    #[inline]
    pub fn is_signed(self) -> bool {
        matches!(self.kind(), Int(_))
    }

    #[inline]
    pub fn is_ptr_sized_integral(self) -> bool {
        matches!(self.kind(), Int(ty::IntTy::Isize) | Uint(ty::UintTy::Usize))
    }

    #[inline]
    pub fn has_concrete_skeleton(self) -> bool {
        !matches!(self.kind(), Param(_) | Infer(_) | Error(_))
    }

    /// Checks whether a type recursively contains another type
    ///
    /// Example: `Option<()>` contains `()`
    pub fn contains(self, other: Ty<'tcx>) -> bool {
        struct ContainsTyVisitor<'tcx>(Ty<'tcx>);

        impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for ContainsTyVisitor<'tcx> {
            type Result = ControlFlow<()>;

            fn visit_ty(&mut self, t: Ty<'tcx>) -> Self::Result {
                if self.0 == t { ControlFlow::Break(()) } else { t.super_visit_with(self) }
            }
        }

        let cf = self.visit_with(&mut ContainsTyVisitor(other));
        cf.is_break()
    }

    /// Checks whether a type recursively contains any closure
    ///
    /// Example: `Option<{closure@file.rs:4:20}>` returns true
    pub fn contains_closure(self) -> bool {
        struct ContainsClosureVisitor;

        impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for ContainsClosureVisitor {
            type Result = ControlFlow<()>;

            fn visit_ty(&mut self, t: Ty<'tcx>) -> Self::Result {
                if let ty::Closure(..) = t.kind() {
                    ControlFlow::Break(())
                } else {
                    t.super_visit_with(self)
                }
            }
        }

        let cf = self.visit_with(&mut ContainsClosureVisitor);
        cf.is_break()
    }

    /// Returns the type and mutability of `*ty`.
    ///
    /// The parameter `explicit` indicates if this is an *explicit* dereference.
    /// Some types -- notably raw ptrs -- can only be dereferenced explicitly.
    pub fn builtin_deref(self, explicit: bool) -> Option<Ty<'tcx>> {
        match *self.kind() {
            _ if let Some(boxed) = self.boxed_ty() => Some(boxed),
            Ref(_, ty, _) => Some(ty),
            RawPtr(ty, _) if explicit => Some(ty),
            _ => None,
        }
    }

    /// Returns the type of `ty[i]`.
    pub fn builtin_index(self) -> Option<Ty<'tcx>> {
        match self.kind() {
            Array(ty, _) | Slice(ty) => Some(*ty),
            _ => None,
        }
    }

    #[tracing::instrument(level = "trace", skip(tcx))]
    pub fn fn_sig(self, tcx: TyCtxt<'tcx>) -> PolyFnSig<'tcx> {
        self.kind().fn_sig(tcx)
    }

    #[inline]
    pub fn is_fn(self) -> bool {
        matches!(self.kind(), FnDef(..) | FnPtr(..))
    }

    #[inline]
    pub fn is_fn_ptr(self) -> bool {
        matches!(self.kind(), FnPtr(..))
    }

    #[inline]
    pub fn is_impl_trait(self) -> bool {
        matches!(self.kind(), Alias(ty::Opaque, ..))
    }

    #[inline]
    pub fn ty_adt_def(self) -> Option<AdtDef<'tcx>> {
        match self.kind() {
            Adt(adt, _) => Some(*adt),
            _ => None,
        }
    }

    /// Iterates over tuple fields.
    /// Panics when called on anything but a tuple.
    #[inline]
    pub fn tuple_fields(self) -> &'tcx List<Ty<'tcx>> {
        match self.kind() {
            Tuple(args) => args,
            _ => bug!("tuple_fields called on non-tuple: {self:?}"),
        }
    }

    /// If the type contains variants, returns the valid range of variant indices.
    //
    // FIXME: This requires the optimized MIR in the case of coroutines.
    #[inline]
    pub fn variant_range(self, tcx: TyCtxt<'tcx>) -> Option<Range<VariantIdx>> {
        match self.kind() {
            TyKind::Adt(adt, _) => Some(adt.variant_range()),
            TyKind::Coroutine(def_id, args) => {
                Some(args.as_coroutine().variant_range(*def_id, tcx))
            }
            _ => None,
        }
    }

    /// If the type contains variants, returns the variant for `variant_index`.
    /// Panics if `variant_index` is out of range.
    //
    // FIXME: This requires the optimized MIR in the case of coroutines.
    #[inline]
    pub fn discriminant_for_variant(
        self,
        tcx: TyCtxt<'tcx>,
        variant_index: VariantIdx,
    ) -> Option<Discr<'tcx>> {
        match self.kind() {
            TyKind::Adt(adt, _) if adt.is_enum() => {
                Some(adt.discriminant_for_variant(tcx, variant_index))
            }
            TyKind::Coroutine(def_id, args) => {
                Some(args.as_coroutine().discriminant_for_variant(*def_id, tcx, variant_index))
            }
            _ => None,
        }
    }

    /// Returns the type of the discriminant of this type.
    pub fn discriminant_ty(self, tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        match self.kind() {
            ty::Adt(adt, _) if adt.is_enum() => adt.repr().discr_type().to_ty(tcx),
            ty::Coroutine(_, args) => args.as_coroutine().discr_ty(tcx),

            ty::Param(_) | ty::Alias(..) | ty::Infer(ty::TyVar(_)) => {
                let assoc_items = tcx.associated_item_def_ids(
                    tcx.require_lang_item(hir::LangItem::DiscriminantKind, None),
                );
                Ty::new_projection_from_args(tcx, assoc_items[0], tcx.mk_args(&[self.into()]))
            }

            ty::Pat(ty, _) => ty.discriminant_ty(tcx),

            ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Adt(..)
            | ty::Foreign(_)
            | ty::Str
            | ty::Array(..)
            | ty::Slice(_)
            | ty::RawPtr(_, _)
            | ty::Ref(..)
            | ty::FnDef(..)
            | ty::FnPtr(..)
            | ty::Dynamic(..)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::CoroutineWitness(..)
            | ty::Never
            | ty::Tuple(_)
            | ty::UnsafeBinder(_)
            | ty::Error(_)
            | ty::Infer(IntVar(_) | FloatVar(_)) => tcx.types.u8,

            ty::Bound(..)
            | ty::Placeholder(_)
            | ty::Infer(FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) => {
                bug!("`discriminant_ty` applied to unexpected type: {:?}", self)
            }
        }
    }

    /// Returns the type of the async destructor of this type.
    pub fn async_destructor_ty(self, tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        match self.async_drop_glue_morphology(tcx) {
            AsyncDropGlueMorphology::Noop => {
                return Ty::async_destructor_combinator(tcx, LangItem::AsyncDropNoop)
                    .instantiate_identity();
            }
            AsyncDropGlueMorphology::DeferredDropInPlace => {
                let drop_in_place =
                    Ty::async_destructor_combinator(tcx, LangItem::AsyncDropDeferredDropInPlace)
                        .instantiate(tcx, &[self.into()]);
                return Ty::async_destructor_combinator(tcx, LangItem::AsyncDropFuse)
                    .instantiate(tcx, &[drop_in_place.into()]);
            }
            AsyncDropGlueMorphology::Custom => (),
        }

        match *self.kind() {
            ty::Param(_) | ty::Alias(..) | ty::Infer(ty::TyVar(_)) => {
                let assoc_items = tcx
                    .associated_item_def_ids(tcx.require_lang_item(LangItem::AsyncDestruct, None));
                Ty::new_projection(tcx, assoc_items[0], [self])
            }

            ty::Array(elem_ty, _) | ty::Slice(elem_ty) => {
                let dtor = Ty::async_destructor_combinator(tcx, LangItem::AsyncDropSlice)
                    .instantiate(tcx, &[elem_ty.into()]);
                Ty::async_destructor_combinator(tcx, LangItem::AsyncDropFuse)
                    .instantiate(tcx, &[dtor.into()])
            }

            ty::Adt(adt_def, args) if adt_def.is_enum() || adt_def.is_struct() => self
                .adt_async_destructor_ty(
                    tcx,
                    adt_def.variants().iter().map(|v| v.fields.iter().map(|f| f.ty(tcx, args))),
                ),
            ty::Tuple(tys) => self.adt_async_destructor_ty(tcx, iter::once(tys)),
            ty::Closure(_, args) => {
                self.adt_async_destructor_ty(tcx, iter::once(args.as_closure().upvar_tys()))
            }
            ty::CoroutineClosure(_, args) => self
                .adt_async_destructor_ty(tcx, iter::once(args.as_coroutine_closure().upvar_tys())),

            ty::Adt(adt_def, _) => {
                assert!(adt_def.is_union());

                let surface_drop = self.surface_async_dropper_ty(tcx).unwrap();

                Ty::async_destructor_combinator(tcx, LangItem::AsyncDropFuse)
                    .instantiate(tcx, &[surface_drop.into()])
            }

            ty::Bound(..)
            | ty::Foreign(_)
            | ty::Placeholder(_)
            | ty::Infer(ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) => {
                bug!("`async_destructor_ty` applied to unexpected type: {self:?}")
            }

            _ => bug!("`async_destructor_ty` is not yet implemented for type: {self:?}"),
        }
    }

    fn adt_async_destructor_ty<I>(self, tcx: TyCtxt<'tcx>, variants: I) -> Ty<'tcx>
    where
        I: Iterator + ExactSizeIterator,
        I::Item: IntoIterator<Item = Ty<'tcx>>,
    {
        debug_assert_eq!(self.async_drop_glue_morphology(tcx), AsyncDropGlueMorphology::Custom);

        let defer = Ty::async_destructor_combinator(tcx, LangItem::AsyncDropDefer);
        let chain = Ty::async_destructor_combinator(tcx, LangItem::AsyncDropChain);

        let noop =
            Ty::async_destructor_combinator(tcx, LangItem::AsyncDropNoop).instantiate_identity();
        let either = Ty::async_destructor_combinator(tcx, LangItem::AsyncDropEither);

        let variants_dtor = variants
            .into_iter()
            .map(|variant| {
                variant
                    .into_iter()
                    .map(|ty| defer.instantiate(tcx, &[ty.into()]))
                    .reduce(|acc, next| chain.instantiate(tcx, &[acc.into(), next.into()]))
                    .unwrap_or(noop)
            })
            .reduce(|other, matched| {
                either.instantiate(tcx, &[other.into(), matched.into(), self.into()])
            })
            .unwrap();

        let dtor = if let Some(dropper_ty) = self.surface_async_dropper_ty(tcx) {
            Ty::async_destructor_combinator(tcx, LangItem::AsyncDropChain)
                .instantiate(tcx, &[dropper_ty.into(), variants_dtor.into()])
        } else {
            variants_dtor
        };

        Ty::async_destructor_combinator(tcx, LangItem::AsyncDropFuse)
            .instantiate(tcx, &[dtor.into()])
    }

    fn surface_async_dropper_ty(self, tcx: TyCtxt<'tcx>) -> Option<Ty<'tcx>> {
        let adt_def = self.ty_adt_def()?;
        let dropper = adt_def
            .async_destructor(tcx)
            .map(|_| LangItem::SurfaceAsyncDropInPlace)
            .or_else(|| adt_def.destructor(tcx).map(|_| LangItem::AsyncDropSurfaceDropInPlace))?;
        Some(Ty::async_destructor_combinator(tcx, dropper).instantiate(tcx, &[self.into()]))
    }

    fn async_destructor_combinator(
        tcx: TyCtxt<'tcx>,
        lang_item: LangItem,
    ) -> ty::EarlyBinder<'tcx, Ty<'tcx>> {
        tcx.fn_sig(tcx.require_lang_item(lang_item, None))
            .map_bound(|fn_sig| fn_sig.output().no_bound_vars().unwrap())
    }

    /// Returns the type of metadata for (potentially wide) pointers to this type,
    /// or the struct tail if the metadata type cannot be determined.
    pub fn ptr_metadata_ty_or_tail(
        self,
        tcx: TyCtxt<'tcx>,
        normalize: impl FnMut(Ty<'tcx>) -> Ty<'tcx>,
    ) -> Result<Ty<'tcx>, Ty<'tcx>> {
        let tail = tcx.struct_tail_raw(self, normalize, || {});
        match tail.kind() {
            // Sized types
            ty::Infer(ty::IntVar(_) | ty::FloatVar(_))
            | ty::Uint(_)
            | ty::Int(_)
            | ty::Bool
            | ty::Float(_)
            | ty::FnDef(..)
            | ty::FnPtr(..)
            | ty::RawPtr(..)
            | ty::Char
            | ty::Ref(..)
            | ty::Coroutine(..)
            | ty::CoroutineWitness(..)
            | ty::Array(..)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Never
            | ty::Error(_)
            // Extern types have metadata = ().
            | ty::Foreign(..)
            // `dyn*` has metadata = ().
            | ty::Dynamic(_, _, ty::DynStar)
            // If returned by `struct_tail_raw` this is a unit struct
            // without any fields, or not a struct, and therefore is Sized.
            | ty::Adt(..)
            // If returned by `struct_tail_raw` this is the empty tuple,
            // a.k.a. unit type, which is Sized
            | ty::Tuple(..) => Ok(tcx.types.unit),

            ty::Str | ty::Slice(_) => Ok(tcx.types.usize),

            ty::Dynamic(_, _, ty::Dyn) => {
                let dyn_metadata = tcx.require_lang_item(LangItem::DynMetadata, None);
                Ok(tcx.type_of(dyn_metadata).instantiate(tcx, &[tail.into()]))
            }

            // We don't know the metadata of `self`, but it must be equal to the
            // metadata of `tail`.
            ty::Param(_) | ty::Alias(..) => Err(tail),

            | ty::UnsafeBinder(_) => todo!("FIXME(unsafe_binder)"),

            ty::Infer(ty::TyVar(_))
            | ty::Pat(..)
            | ty::Bound(..)
            | ty::Placeholder(..)
            | ty::Infer(ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) => bug!(
                "`ptr_metadata_ty_or_tail` applied to unexpected type: {self:?} (tail = {tail:?})"
            ),
        }
    }

    /// Returns the type of metadata for (potentially wide) pointers to this type.
    /// Causes an ICE if the metadata type cannot be determined.
    pub fn ptr_metadata_ty(
        self,
        tcx: TyCtxt<'tcx>,
        normalize: impl FnMut(Ty<'tcx>) -> Ty<'tcx>,
    ) -> Ty<'tcx> {
        match self.ptr_metadata_ty_or_tail(tcx, normalize) {
            Ok(metadata) => metadata,
            Err(tail) => bug!(
                "`ptr_metadata_ty` failed to get metadata for type: {self:?} (tail = {tail:?})"
            ),
        }
    }

    /// Given a pointer or reference type, returns the type of the *pointee*'s
    /// metadata. If it can't be determined exactly (perhaps due to still
    /// being generic) then a projection through `ptr::Pointee` will be returned.
    ///
    /// This is particularly useful for getting the type of the result of
    /// [`UnOp::PtrMetadata`](crate::mir::UnOp::PtrMetadata).
    ///
    /// Panics if `self` is not dereferencable.
    #[track_caller]
    pub fn pointee_metadata_ty_or_projection(self, tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        let Some(pointee_ty) = self.builtin_deref(true) else {
            bug!("Type {self:?} is not a pointer or reference type")
        };
        if pointee_ty.is_trivially_sized(tcx) {
            tcx.types.unit
        } else {
            match pointee_ty.ptr_metadata_ty_or_tail(tcx, |x| x) {
                Ok(metadata_ty) => metadata_ty,
                Err(tail_ty) => {
                    let Some(metadata_def_id) = tcx.lang_items().metadata_type() else {
                        bug!("No metadata_type lang item while looking at {self:?}")
                    };
                    Ty::new_projection(tcx, metadata_def_id, [tail_ty])
                }
            }
        }
    }

    /// When we create a closure, we record its kind (i.e., what trait
    /// it implements, constrained by how it uses its borrows) into its
    /// [`ty::ClosureArgs`] or [`ty::CoroutineClosureArgs`] using a type
    /// parameter. This is kind of a phantom type, except that the
    /// most convenient thing for us to are the integral types. This
    /// function converts such a special type into the closure
    /// kind. To go the other way, use [`Ty::from_closure_kind`].
    ///
    /// Note that during type checking, we use an inference variable
    /// to represent the closure kind, because it has not yet been
    /// inferred. Once upvar inference (in `rustc_hir_analysis/src/check/upvar.rs`)
    /// is complete, that type variable will be unified with one of
    /// the integral types.
    ///
    /// ```rust,ignore (snippet of compiler code)
    /// if let TyKind::Closure(def_id, args) = closure_ty.kind()
    ///     && let Some(closure_kind) = args.as_closure().kind_ty().to_opt_closure_kind()
    /// {
    ///     println!("{closure_kind:?}");
    /// } else if let TyKind::CoroutineClosure(def_id, args) = closure_ty.kind()
    ///     && let Some(closure_kind) = args.as_coroutine_closure().kind_ty().to_opt_closure_kind()
    /// {
    ///     println!("{closure_kind:?}");
    /// }
    /// ```
    ///
    /// After upvar analysis, you should instead use [`ty::ClosureArgs::kind()`]
    /// or [`ty::CoroutineClosureArgs::kind()`] to assert that the `ClosureKind`
    /// has been constrained instead of manually calling this method.
    ///
    /// ```rust,ignore (snippet of compiler code)
    /// if let TyKind::Closure(def_id, args) = closure_ty.kind()
    /// {
    ///     println!("{:?}", args.as_closure().kind());
    /// } else if let TyKind::CoroutineClosure(def_id, args) = closure_ty.kind()
    /// {
    ///     println!("{:?}", args.as_coroutine_closure().kind());
    /// }
    /// ```
    pub fn to_opt_closure_kind(self) -> Option<ty::ClosureKind> {
        match self.kind() {
            Int(int_ty) => match int_ty {
                ty::IntTy::I8 => Some(ty::ClosureKind::Fn),
                ty::IntTy::I16 => Some(ty::ClosureKind::FnMut),
                ty::IntTy::I32 => Some(ty::ClosureKind::FnOnce),
                _ => bug!("cannot convert type `{:?}` to a closure kind", self),
            },

            // "Bound" types appear in canonical queries when the
            // closure type is not yet known, and `Placeholder` and `Param`
            // may be encountered in generic `AsyncFnKindHelper` goals.
            Bound(..) | Placeholder(_) | Param(_) | Infer(_) => None,

            Error(_) => Some(ty::ClosureKind::Fn),

            _ => bug!("cannot convert type `{:?}` to a closure kind", self),
        }
    }

    /// Inverse of [`Ty::to_opt_closure_kind`]. See docs on that method
    /// for explanation of the relationship between `Ty` and [`ty::ClosureKind`].
    pub fn from_closure_kind(tcx: TyCtxt<'tcx>, kind: ty::ClosureKind) -> Ty<'tcx> {
        match kind {
            ty::ClosureKind::Fn => tcx.types.i8,
            ty::ClosureKind::FnMut => tcx.types.i16,
            ty::ClosureKind::FnOnce => tcx.types.i32,
        }
    }

    /// Like [`Ty::to_opt_closure_kind`], but it caps the "maximum" closure kind
    /// to `FnMut`. This is because although we have three capability states,
    /// `AsyncFn`/`AsyncFnMut`/`AsyncFnOnce`, we only need to distinguish two coroutine
    /// bodies: by-ref and by-value.
    ///
    /// See the definition of `AsyncFn` and `AsyncFnMut` and the `CallRefFuture`
    /// associated type for why we don't distinguish [`ty::ClosureKind::Fn`] and
    /// [`ty::ClosureKind::FnMut`] for the purpose of the generated MIR bodies.
    ///
    /// This method should be used when constructing a `Coroutine` out of a
    /// `CoroutineClosure`, when the `Coroutine`'s `kind` field is being populated
    /// directly from the `CoroutineClosure`'s `kind`.
    pub fn from_coroutine_closure_kind(tcx: TyCtxt<'tcx>, kind: ty::ClosureKind) -> Ty<'tcx> {
        match kind {
            ty::ClosureKind::Fn | ty::ClosureKind::FnMut => tcx.types.i16,
            ty::ClosureKind::FnOnce => tcx.types.i32,
        }
    }

    /// Fast path helper for testing if a type is `Sized`.
    ///
    /// Returning true means the type is known to be sized. Returning
    /// `false` means nothing -- could be sized, might not be.
    ///
    /// Note that we could never rely on the fact that a type such as `[_]` is
    /// trivially `!Sized` because we could be in a type environment with a
    /// bound such as `[_]: Copy`. A function with such a bound obviously never
    /// can be called, but that doesn't mean it shouldn't typecheck. This is why
    /// this method doesn't return `Option<bool>`.
    pub fn is_trivially_sized(self, tcx: TyCtxt<'tcx>) -> bool {
        match self.kind() {
            ty::Infer(ty::IntVar(_) | ty::FloatVar(_))
            | ty::Uint(_)
            | ty::Int(_)
            | ty::Bool
            | ty::Float(_)
            | ty::FnDef(..)
            | ty::FnPtr(..)
            | ty::UnsafeBinder(_)
            | ty::RawPtr(..)
            | ty::Char
            | ty::Ref(..)
            | ty::Coroutine(..)
            | ty::CoroutineWitness(..)
            | ty::Array(..)
            | ty::Pat(..)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Never
            | ty::Error(_)
            | ty::Dynamic(_, _, ty::DynStar) => true,

            ty::Str | ty::Slice(_) | ty::Dynamic(_, _, ty::Dyn) | ty::Foreign(..) => false,

            ty::Tuple(tys) => tys.last().is_none_or(|ty| ty.is_trivially_sized(tcx)),

            ty::Adt(def, args) => def
                .sized_constraint(tcx)
                .is_none_or(|ty| ty.instantiate(tcx, args).is_trivially_sized(tcx)),

            ty::Alias(..) | ty::Param(_) | ty::Placeholder(..) | ty::Bound(..) => false,

            ty::Infer(ty::TyVar(_)) => false,

            ty::Infer(ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) => {
                bug!("`is_trivially_sized` applied to unexpected type: {:?}", self)
            }
        }
    }

    /// Fast path helper for primitives which are always `Copy` and which
    /// have a side-effect-free `Clone` impl.
    ///
    /// Returning true means the type is known to be pure and `Copy+Clone`.
    /// Returning `false` means nothing -- could be `Copy`, might not be.
    ///
    /// This is mostly useful for optimizations, as these are the types
    /// on which we can replace cloning with dereferencing.
    pub fn is_trivially_pure_clone_copy(self) -> bool {
        match self.kind() {
            ty::Bool | ty::Char | ty::Never => true,

            // These aren't even `Clone`
            ty::Str | ty::Slice(..) | ty::Foreign(..) | ty::Dynamic(..) => false,

            ty::Infer(ty::InferTy::FloatVar(_) | ty::InferTy::IntVar(_))
            | ty::Int(..)
            | ty::Uint(..)
            | ty::Float(..) => true,

            // ZST which can't be named are fine.
            ty::FnDef(..) => true,

            ty::Array(element_ty, _len) => element_ty.is_trivially_pure_clone_copy(),

            // A 100-tuple isn't "trivial", so doing this only for reasonable sizes.
            ty::Tuple(field_tys) => {
                field_tys.len() <= 3 && field_tys.iter().all(Self::is_trivially_pure_clone_copy)
            }

            ty::Pat(ty, _) => ty.is_trivially_pure_clone_copy(),

            // Sometimes traits aren't implemented for every ABI or arity,
            // because we can't be generic over everything yet.
            ty::FnPtr(..) => false,

            // Definitely absolutely not copy.
            ty::Ref(_, _, hir::Mutability::Mut) => false,

            // The standard library has a blanket Copy impl for shared references and raw pointers,
            // for all unsized types.
            ty::Ref(_, _, hir::Mutability::Not) | ty::RawPtr(..) => true,

            ty::Coroutine(..) | ty::CoroutineWitness(..) => false,

            // Might be, but not "trivial" so just giving the safe answer.
            ty::Adt(..) | ty::Closure(..) | ty::CoroutineClosure(..) => false,

            ty::UnsafeBinder(_) => false,

            // Needs normalization or revealing to determine, so no is the safe answer.
            ty::Alias(..) => false,

            ty::Param(..) | ty::Infer(..) | ty::Error(..) => false,

            ty::Bound(..) | ty::Placeholder(..) => {
                bug!("`is_trivially_pure_clone_copy` applied to unexpected type: {:?}", self);
            }
        }
    }

    /// If `self` is a primitive, return its [`Symbol`].
    pub fn primitive_symbol(self) -> Option<Symbol> {
        match self.kind() {
            ty::Bool => Some(sym::bool),
            ty::Char => Some(sym::char),
            ty::Float(f) => match f {
                ty::FloatTy::F16 => Some(sym::f16),
                ty::FloatTy::F32 => Some(sym::f32),
                ty::FloatTy::F64 => Some(sym::f64),
                ty::FloatTy::F128 => Some(sym::f128),
            },
            ty::Int(f) => match f {
                ty::IntTy::Isize => Some(sym::isize),
                ty::IntTy::I8 => Some(sym::i8),
                ty::IntTy::I16 => Some(sym::i16),
                ty::IntTy::I32 => Some(sym::i32),
                ty::IntTy::I64 => Some(sym::i64),
                ty::IntTy::I128 => Some(sym::i128),
            },
            ty::Uint(f) => match f {
                ty::UintTy::Usize => Some(sym::usize),
                ty::UintTy::U8 => Some(sym::u8),
                ty::UintTy::U16 => Some(sym::u16),
                ty::UintTy::U32 => Some(sym::u32),
                ty::UintTy::U64 => Some(sym::u64),
                ty::UintTy::U128 => Some(sym::u128),
            },
            ty::Str => Some(sym::str),
            _ => None,
        }
    }

    pub fn is_c_void(self, tcx: TyCtxt<'_>) -> bool {
        match self.kind() {
            ty::Adt(adt, _) => tcx.is_lang_item(adt.did(), LangItem::CVoid),
            _ => false,
        }
    }

    /// Returns `true` when the outermost type cannot be further normalized,
    /// resolved, or instantiated. This includes all primitive types, but also
    /// things like ADTs and trait objects, since even if their arguments or
    /// nested types may be further simplified, the outermost [`TyKind`] or
    /// type constructor remains the same.
    pub fn is_known_rigid(self) -> bool {
        self.kind().is_known_rigid()
    }
}

impl<'tcx> rustc_type_ir::inherent::Tys<TyCtxt<'tcx>> for &'tcx ty::List<Ty<'tcx>> {
    fn inputs(self) -> &'tcx [Ty<'tcx>] {
        self.split_last().unwrap().1
    }

    fn output(self) -> Ty<'tcx> {
        *self.split_last().unwrap().0
    }
}

// Some types are used a lot. Make sure they don't unintentionally get bigger.
#[cfg(target_pointer_width = "64")]
mod size_asserts {
    use rustc_data_structures::static_assert_size;

    use super::*;
    // tidy-alphabetical-start
    static_assert_size!(ty::RegionKind<'_>, 24);
    static_assert_size!(ty::TyKind<'_>, 24);
    // tidy-alphabetical-end
}
