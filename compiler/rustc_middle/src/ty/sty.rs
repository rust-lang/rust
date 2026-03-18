//! This module contains `TyKind` and its major components.

#![allow(rustc::usage_of_ty_tykind)]

use std::borrow::Cow;
use std::ops::Range;

use rustc_abi::VariantIdx;
use rustc_hir::def_id::DefId;
use rustc_macros::{HashStable, TyDecodable, TyEncodable, TypeFoldable, extension};
use rustc_span::{Span, Symbol, kw};
use rustc_type_ir::TyKind::*;
use rustc_type_ir::{self as ir, TypeVisitableExt};
use tracing::instrument;

use crate::infer::canonical::Canonical;
use crate::ty::{self, Discr, GenericArg, GenericArgsRef, List, ParamEnv, TyCtxt, TypeVisitable};

// Re-export and re-parameterize some `I = TyCtxt<'tcx>` types here
#[rustc_diagnostic_item = "TyKind"]
pub type TyKind<'tcx> = ir::TyKind<TyCtxt<'tcx>>;
pub type TypeAndMut<'tcx> = ir::TypeAndMut<TyCtxt<'tcx>>;
pub type AliasTy<'tcx> = ir::AliasTy<TyCtxt<'tcx>>;
pub type FnSig<'tcx> = ir::FnSig<TyCtxt<'tcx>>;
pub type Binder<'tcx, T> = ir::Binder<TyCtxt<'tcx>, T>;
pub type EarlyBinder<'tcx, T> = ir::EarlyBinder<TyCtxt<'tcx>, T>;
pub type TypingMode<'tcx> = ir::TypingMode<TyCtxt<'tcx>>;
pub type Placeholder<'tcx, T> = ir::Placeholder<TyCtxt<'tcx>, T>;
pub type PlaceholderRegion<'tcx> = ir::PlaceholderRegion<TyCtxt<'tcx>>;
pub type PlaceholderType<'tcx> = ir::PlaceholderType<TyCtxt<'tcx>>;
pub type PlaceholderConst<'tcx> = ir::PlaceholderConst<TyCtxt<'tcx>>;
pub type BoundTy<'tcx> = ir::BoundTy<TyCtxt<'tcx>>;
pub type BoundConst<'tcx> = ir::BoundConst<TyCtxt<'tcx>>;
pub type BoundRegion<'tcx> = ir::BoundRegion<TyCtxt<'tcx>>;
pub type BoundVariableKind<'tcx> = ir::BoundVariableKind<TyCtxt<'tcx>>;
pub type BoundRegionKind<'tcx> = ir::BoundRegionKind<TyCtxt<'tcx>>;
pub type BoundTyKind<'tcx> = ir::BoundTyKind<TyCtxt<'tcx>>;
#[allow(rustc::usage_of_qualified_ty)]
pub type Ty<'tcx> = ir::Ty<TyCtxt<'tcx>>;

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
        tcx.coroutine_variant_range(def_id, self.args.as_coroutine())
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
        let layout = tcx.coroutine_layout(def_id, self.args).unwrap();
        layout.variant_fields.iter().map(move |variant| {
            variant.iter().map(move |field| {
                if tcx.is_async_drop_in_place_coroutine(def_id) {
                    layout.field_tys[*field].ty
                } else {
                    ty::EarlyBinder::bind(layout.field_tys[*field].ty).instantiate(tcx, self.args)
                }
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
        Ty::new_param(tcx, ParamTy::new(self.index, self.name))
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
    pub fn find_const_ty_from_env<'tcx>(self, env: ParamEnv<'tcx>) -> Ty<'tcx> {
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

        // N.B. it may be tempting to fix ICEs by making this function return
        // `Option<Ty<'tcx>>` instead of `Ty<'tcx>`; however, this is generally
        // considered to be a bandaid solution, since it hides more important
        // underlying issues with how we construct generics and predicates of
        // items. It's advised to fix the underlying issue rather than trying
        // to modify this function.
        let ty = candidates.next().unwrap_or_else(|| {
            bug!("cannot find `{self:?}` in param-env: {env:#?}");
        });
        assert!(
            candidates.next().is_none(),
            "did not expect duplicate `ConstParamHasTy` for `{self:?}` in param-env: {env:#?}"
        );
        ty
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

impl<'tcx> rustc_type_ir::inherent::Symbol<TyCtxt<'tcx>> for Symbol {
    fn is_kw_underscore_lifetime(self) -> bool {
        self == kw::UnderscoreLifetime
    }
}

// Some types are used a lot. Make sure they don't unintentionally get bigger.
#[cfg(target_pointer_width = "64")]
mod size_asserts {
    use rustc_data_structures::static_assert_size;

    use super::*;
    // tidy-alphabetical-start
    static_assert_size!(TyKind<'_>, 24);
    static_assert_size!(ty::WithCachedTypeInfo<TyKind<'_>>, 48);
    // tidy-alphabetical-end
}
