use rustc_data_structures::intern::Interned;
use rustc_errors::MultiSpan;
use rustc_hir::def_id::DefId;
use rustc_macros::{HashStable, TyDecodable, TyEncodable};
use rustc_span::{DUMMY_SP, ErrorGuaranteed, Symbol, kw, sym};
use rustc_type_ir::RegionKind as IrRegionKind;
pub use rustc_type_ir::RegionVid;
use tracing::debug;

use crate::ty::{self, BoundVar, TyCtxt, TypeFlags};

pub type RegionKind<'tcx> = IrRegionKind<TyCtxt<'tcx>>;

/// Use this rather than `RegionKind`, whenever possible.
#[derive(Copy, Clone, PartialEq, Eq, Hash, HashStable)]
#[rustc_pass_by_value]
pub struct Region<'tcx>(pub Interned<'tcx, RegionKind<'tcx>>);

impl<'tcx> rustc_type_ir::inherent::IntoKind for Region<'tcx> {
    type Kind = RegionKind<'tcx>;

    fn kind(self) -> RegionKind<'tcx> {
        *self.0.0
    }
}

impl<'tcx> rustc_type_ir::Flags for Region<'tcx> {
    fn flags(&self) -> TypeFlags {
        self.type_flags()
    }

    fn outer_exclusive_binder(&self) -> ty::DebruijnIndex {
        match self.kind() {
            ty::ReBound(debruijn, _) => debruijn.shifted_in(1),
            _ => ty::INNERMOST,
        }
    }
}

impl<'tcx> Region<'tcx> {
    #[inline]
    pub fn new_early_param(
        tcx: TyCtxt<'tcx>,
        early_bound_region: ty::EarlyParamRegion,
    ) -> Region<'tcx> {
        tcx.intern_region(ty::ReEarlyParam(early_bound_region))
    }

    #[inline]
    pub fn new_bound(
        tcx: TyCtxt<'tcx>,
        debruijn: ty::DebruijnIndex,
        bound_region: ty::BoundRegion,
    ) -> Region<'tcx> {
        // Use a pre-interned one when possible.
        if let ty::BoundRegion { var, kind: ty::BoundRegionKind::Anon } = bound_region
            && let Some(inner) = tcx.lifetimes.re_late_bounds.get(debruijn.as_usize())
            && let Some(re) = inner.get(var.as_usize()).copied()
        {
            re
        } else {
            tcx.intern_region(ty::ReBound(debruijn, bound_region))
        }
    }

    #[inline]
    pub fn new_late_param(
        tcx: TyCtxt<'tcx>,
        scope: DefId,
        kind: LateParamRegionKind,
    ) -> Region<'tcx> {
        let data = LateParamRegion { scope, kind };
        tcx.intern_region(ty::ReLateParam(data))
    }

    #[inline]
    pub fn new_var(tcx: TyCtxt<'tcx>, v: ty::RegionVid) -> Region<'tcx> {
        // Use a pre-interned one when possible.
        tcx.lifetimes
            .re_vars
            .get(v.as_usize())
            .copied()
            .unwrap_or_else(|| tcx.intern_region(ty::ReVar(v)))
    }

    #[inline]
    pub fn new_placeholder(tcx: TyCtxt<'tcx>, placeholder: ty::PlaceholderRegion) -> Region<'tcx> {
        tcx.intern_region(ty::RePlaceholder(placeholder))
    }

    /// Constructs a `RegionKind::ReError` region.
    #[track_caller]
    pub fn new_error(tcx: TyCtxt<'tcx>, guar: ErrorGuaranteed) -> Region<'tcx> {
        tcx.intern_region(ty::ReError(guar))
    }

    /// Constructs a `RegionKind::ReError` region and registers a delayed bug to ensure it gets
    /// used.
    #[track_caller]
    pub fn new_error_misc(tcx: TyCtxt<'tcx>) -> Region<'tcx> {
        Region::new_error_with_message(
            tcx,
            DUMMY_SP,
            "RegionKind::ReError constructed but no error reported",
        )
    }

    /// Constructs a `RegionKind::ReError` region and registers a delayed bug with the given `msg`
    /// to ensure it gets used.
    #[track_caller]
    pub fn new_error_with_message<S: Into<MultiSpan>>(
        tcx: TyCtxt<'tcx>,
        span: S,
        msg: &'static str,
    ) -> Region<'tcx> {
        let reported = tcx.dcx().span_delayed_bug(span, msg);
        Region::new_error(tcx, reported)
    }

    /// Avoid this in favour of more specific `new_*` methods, where possible,
    /// to avoid the cost of the `match`.
    pub fn new_from_kind(tcx: TyCtxt<'tcx>, kind: RegionKind<'tcx>) -> Region<'tcx> {
        match kind {
            ty::ReEarlyParam(region) => Region::new_early_param(tcx, region),
            ty::ReBound(debruijn, region) => Region::new_bound(tcx, debruijn, region),
            ty::ReLateParam(ty::LateParamRegion { scope, kind }) => {
                Region::new_late_param(tcx, scope, kind)
            }
            ty::ReStatic => tcx.lifetimes.re_static,
            ty::ReVar(vid) => Region::new_var(tcx, vid),
            ty::RePlaceholder(region) => Region::new_placeholder(tcx, region),
            ty::ReErased => tcx.lifetimes.re_erased,
            ty::ReError(reported) => Region::new_error(tcx, reported),
        }
    }
}

impl<'tcx> rustc_type_ir::inherent::Region<TyCtxt<'tcx>> for Region<'tcx> {
    fn new_bound(
        interner: TyCtxt<'tcx>,
        debruijn: ty::DebruijnIndex,
        var: ty::BoundRegion,
    ) -> Self {
        Region::new_bound(interner, debruijn, var)
    }

    fn new_anon_bound(tcx: TyCtxt<'tcx>, debruijn: ty::DebruijnIndex, var: ty::BoundVar) -> Self {
        Region::new_bound(tcx, debruijn, ty::BoundRegion { var, kind: ty::BoundRegionKind::Anon })
    }

    fn new_static(tcx: TyCtxt<'tcx>) -> Self {
        tcx.lifetimes.re_static
    }
}

/// Region utilities
impl<'tcx> Region<'tcx> {
    pub fn kind(self) -> RegionKind<'tcx> {
        *self.0.0
    }

    pub fn get_name(self) -> Option<Symbol> {
        if self.has_name() {
            match self.kind() {
                ty::ReEarlyParam(ebr) => Some(ebr.name),
                ty::ReBound(_, br) => br.kind.get_name(),
                ty::ReLateParam(fr) => fr.kind.get_name(),
                ty::ReStatic => Some(kw::StaticLifetime),
                ty::RePlaceholder(placeholder) => placeholder.bound.kind.get_name(),
                _ => None,
            }
        } else {
            None
        }
    }

    pub fn get_name_or_anon(self) -> Symbol {
        match self.get_name() {
            Some(name) => name,
            None => sym::anon,
        }
    }

    /// Is this region named by the user?
    pub fn has_name(self) -> bool {
        match self.kind() {
            ty::ReEarlyParam(ebr) => ebr.has_name(),
            ty::ReBound(_, br) => br.kind.is_named(),
            ty::ReLateParam(fr) => fr.kind.is_named(),
            ty::ReStatic => true,
            ty::ReVar(..) => false,
            ty::RePlaceholder(placeholder) => placeholder.bound.kind.is_named(),
            ty::ReErased => false,
            ty::ReError(_) => false,
        }
    }

    #[inline]
    pub fn is_error(self) -> bool {
        matches!(self.kind(), ty::ReError(_))
    }

    #[inline]
    pub fn is_static(self) -> bool {
        matches!(self.kind(), ty::ReStatic)
    }

    #[inline]
    pub fn is_erased(self) -> bool {
        matches!(self.kind(), ty::ReErased)
    }

    #[inline]
    pub fn is_bound(self) -> bool {
        matches!(self.kind(), ty::ReBound(..))
    }

    #[inline]
    pub fn is_placeholder(self) -> bool {
        matches!(self.kind(), ty::RePlaceholder(..))
    }

    #[inline]
    pub fn bound_at_or_above_binder(self, index: ty::DebruijnIndex) -> bool {
        match self.kind() {
            ty::ReBound(debruijn, _) => debruijn >= index,
            _ => false,
        }
    }

    pub fn type_flags(self) -> TypeFlags {
        let mut flags = TypeFlags::empty();

        match self.kind() {
            ty::ReVar(..) => {
                flags = flags | TypeFlags::HAS_FREE_REGIONS;
                flags = flags | TypeFlags::HAS_FREE_LOCAL_REGIONS;
                flags = flags | TypeFlags::HAS_RE_INFER;
            }
            ty::RePlaceholder(..) => {
                flags = flags | TypeFlags::HAS_FREE_REGIONS;
                flags = flags | TypeFlags::HAS_FREE_LOCAL_REGIONS;
                flags = flags | TypeFlags::HAS_RE_PLACEHOLDER;
            }
            ty::ReEarlyParam(..) => {
                flags = flags | TypeFlags::HAS_FREE_REGIONS;
                flags = flags | TypeFlags::HAS_FREE_LOCAL_REGIONS;
                flags = flags | TypeFlags::HAS_RE_PARAM;
            }
            ty::ReLateParam { .. } => {
                flags = flags | TypeFlags::HAS_FREE_REGIONS;
                flags = flags | TypeFlags::HAS_FREE_LOCAL_REGIONS;
            }
            ty::ReStatic => {
                flags = flags | TypeFlags::HAS_FREE_REGIONS;
            }
            ty::ReBound(..) => {
                flags = flags | TypeFlags::HAS_RE_BOUND;
            }
            ty::ReErased => {
                flags = flags | TypeFlags::HAS_RE_ERASED;
            }
            ty::ReError(_) => {
                flags = flags | TypeFlags::HAS_FREE_REGIONS;
                flags = flags | TypeFlags::HAS_ERROR;
            }
        }

        debug!("type_flags({:?}) = {:?}", self, flags);

        flags
    }

    /// True for free regions other than `'static`.
    pub fn is_param(self) -> bool {
        matches!(self.kind(), ty::ReEarlyParam(_) | ty::ReLateParam(_))
    }

    /// True for free region in the current context.
    ///
    /// This is the case for `'static` and param regions.
    pub fn is_free(self) -> bool {
        match self.kind() {
            ty::ReStatic | ty::ReEarlyParam(..) | ty::ReLateParam(..) => true,
            ty::ReVar(..)
            | ty::RePlaceholder(..)
            | ty::ReBound(..)
            | ty::ReErased
            | ty::ReError(..) => false,
        }
    }

    pub fn is_var(self) -> bool {
        matches!(self.kind(), ty::ReVar(_))
    }

    pub fn as_var(self) -> RegionVid {
        match self.kind() {
            ty::ReVar(vid) => vid,
            _ => bug!("expected region {:?} to be of kind ReVar", self),
        }
    }

    /// Given some item `binding_item`, check if this region is a generic parameter introduced by it
    /// or one of the parent generics. Returns the `DefId` of the parameter definition if so.
    pub fn opt_param_def_id(self, tcx: TyCtxt<'tcx>, binding_item: DefId) -> Option<DefId> {
        match self.kind() {
            ty::ReEarlyParam(ebr) => {
                Some(tcx.generics_of(binding_item).region_param(ebr, tcx).def_id)
            }
            ty::ReLateParam(ty::LateParamRegion {
                kind: ty::LateParamRegionKind::Named(def_id, _),
                ..
            }) => Some(def_id),
            _ => None,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, TyEncodable, TyDecodable)]
#[derive(HashStable)]
pub struct EarlyParamRegion {
    pub index: u32,
    pub name: Symbol,
}

impl rustc_type_ir::inherent::ParamLike for EarlyParamRegion {
    fn index(self) -> u32 {
        self.index
    }
}

impl std::fmt::Debug for EarlyParamRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/#{}", self.name, self.index)
    }
}

#[derive(Clone, PartialEq, Eq, Hash, TyEncodable, TyDecodable, Copy)]
#[derive(HashStable)]
/// The parameter representation of late-bound function parameters, "some region
/// at least as big as the scope `fr.scope`".
///
/// Similar to a placeholder region as we create `LateParam` regions when entering a binder
/// except they are always in the root universe and instead of using a boundvar to distinguish
/// between others we use the `DefId` of the parameter. For this reason the `bound_region` field
/// should basically always be `BoundRegionKind::Named` as otherwise there is no way of telling
/// different parameters apart.
pub struct LateParamRegion {
    pub scope: DefId,
    pub kind: LateParamRegionKind,
}

/// When liberating bound regions, we map their [`BoundRegionKind`]
/// to this as we need to track the index of anonymous regions. We
/// otherwise end up liberating multiple bound regions to the same
/// late-bound region.
#[derive(Clone, PartialEq, Eq, Hash, TyEncodable, TyDecodable, Copy)]
#[derive(HashStable)]
pub enum LateParamRegionKind {
    /// An anonymous region parameter for a given fn (&T)
    ///
    /// Unlike [`BoundRegionKind::Anon`], this tracks the index of the
    /// liberated bound region.
    ///
    /// We should ideally never liberate anonymous regions, but do so for the
    /// sake of diagnostics in `FnCtxt::sig_of_closure_with_expectation`.
    Anon(u32),

    /// Named region parameters for functions (a in &'a T)
    ///
    /// The `DefId` is needed to distinguish free regions in
    /// the event of shadowing.
    Named(DefId, Symbol),

    /// Anonymous region for the implicit env pointer parameter
    /// to a closure
    ClosureEnv,
}

impl LateParamRegionKind {
    pub fn from_bound(var: BoundVar, br: BoundRegionKind) -> LateParamRegionKind {
        match br {
            BoundRegionKind::Anon => LateParamRegionKind::Anon(var.as_u32()),
            BoundRegionKind::Named(def_id, name) => LateParamRegionKind::Named(def_id, name),
            BoundRegionKind::ClosureEnv => LateParamRegionKind::ClosureEnv,
        }
    }

    pub fn is_named(&self) -> bool {
        match *self {
            LateParamRegionKind::Named(_, name) => name != kw::UnderscoreLifetime,
            _ => false,
        }
    }

    pub fn get_name(&self) -> Option<Symbol> {
        if self.is_named() {
            match *self {
                LateParamRegionKind::Named(_, name) => return Some(name),
                _ => unreachable!(),
            }
        }

        None
    }

    pub fn get_id(&self) -> Option<DefId> {
        match *self {
            LateParamRegionKind::Named(id, _) => Some(id),
            _ => None,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash, TyEncodable, TyDecodable, Copy)]
#[derive(HashStable)]
pub enum BoundRegionKind {
    /// An anonymous region parameter for a given fn (&T)
    Anon,

    /// Named region parameters for functions (a in &'a T)
    ///
    /// The `DefId` is needed to distinguish free regions in
    /// the event of shadowing.
    Named(DefId, Symbol),

    /// Anonymous region for the implicit env pointer parameter
    /// to a closure
    ClosureEnv,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, TyEncodable, TyDecodable)]
#[derive(HashStable)]
pub struct BoundRegion {
    pub var: BoundVar,
    pub kind: BoundRegionKind,
}

impl<'tcx> rustc_type_ir::inherent::BoundVarLike<TyCtxt<'tcx>> for BoundRegion {
    fn var(self) -> BoundVar {
        self.var
    }

    fn assert_eq(self, var: ty::BoundVariableKind) {
        assert_eq!(self.kind, var.expect_region())
    }
}

impl core::fmt::Debug for BoundRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.kind {
            BoundRegionKind::Anon => write!(f, "{:?}", self.var),
            BoundRegionKind::ClosureEnv => write!(f, "{:?}.Env", self.var),
            BoundRegionKind::Named(def, symbol) => {
                write!(f, "{:?}.Named({:?}, {:?})", self.var, def, symbol)
            }
        }
    }
}

impl BoundRegionKind {
    pub fn is_named(&self) -> bool {
        match *self {
            BoundRegionKind::Named(_, name) => name != kw::UnderscoreLifetime,
            _ => false,
        }
    }

    pub fn get_name(&self) -> Option<Symbol> {
        if self.is_named() {
            match *self {
                BoundRegionKind::Named(_, name) => return Some(name),
                _ => unreachable!(),
            }
        }

        None
    }

    pub fn get_id(&self) -> Option<DefId> {
        match *self {
            BoundRegionKind::Named(id, _) => Some(id),
            _ => None,
        }
    }
}
