use polonius_engine::Atom;
use rustc_data_structures::intern::Interned;
use rustc_errors::MultiSpan;
use rustc_hir::def_id::DefId;
use rustc_index::Idx;
use rustc_span::symbol::sym;
use rustc_span::symbol::{kw, Symbol};
use rustc_span::{ErrorGuaranteed, DUMMY_SP};
use rustc_type_ir::RegionKind as IrRegionKind;
use std::ops::Deref;

use crate::ty::{self, BoundVar, TyCtxt, TypeFlags};

pub type RegionKind<'tcx> = IrRegionKind<TyCtxt<'tcx>>;

/// Use this rather than `RegionKind`, whenever possible.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, HashStable)]
#[rustc_pass_by_value]
pub struct Region<'tcx>(pub Interned<'tcx, RegionKind<'tcx>>);

impl<'tcx> rustc_type_ir::IntoKind for Region<'tcx> {
    type Kind = RegionKind<'tcx>;

    fn kind(self) -> RegionKind<'tcx> {
        *self
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
        if let ty::BoundRegion { var, kind: ty::BrAnon } = bound_region
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
        bound_region: ty::BoundRegionKind,
    ) -> Region<'tcx> {
        tcx.intern_region(ty::ReLateParam(ty::LateParamRegion { scope, bound_region }))
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
    pub fn new_error(tcx: TyCtxt<'tcx>, reported: ErrorGuaranteed) -> Region<'tcx> {
        tcx.intern_region(ty::ReError(reported))
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
            ty::ReLateParam(ty::LateParamRegion { scope, bound_region }) => {
                Region::new_late_param(tcx, scope, bound_region)
            }
            ty::ReStatic => tcx.lifetimes.re_static,
            ty::ReVar(vid) => Region::new_var(tcx, vid),
            ty::RePlaceholder(region) => Region::new_placeholder(tcx, region),
            ty::ReErased => tcx.lifetimes.re_erased,
            ty::ReError(reported) => Region::new_error(tcx, reported),
        }
    }
}

/// Region utilities
impl<'tcx> Region<'tcx> {
    pub fn kind(self) -> RegionKind<'tcx> {
        *self.0.0
    }

    pub fn get_name(self) -> Option<Symbol> {
        if self.has_name() {
            match *self {
                ty::ReEarlyParam(ebr) => Some(ebr.name),
                ty::ReBound(_, br) => br.kind.get_name(),
                ty::ReLateParam(fr) => fr.bound_region.get_name(),
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
        match *self {
            ty::ReEarlyParam(ebr) => ebr.has_name(),
            ty::ReBound(_, br) => br.kind.is_named(),
            ty::ReLateParam(fr) => fr.bound_region.is_named(),
            ty::ReStatic => true,
            ty::ReVar(..) => false,
            ty::RePlaceholder(placeholder) => placeholder.bound.kind.is_named(),
            ty::ReErased => false,
            ty::ReError(_) => false,
        }
    }

    #[inline]
    pub fn is_error(self) -> bool {
        matches!(*self, ty::ReError(_))
    }

    #[inline]
    pub fn is_static(self) -> bool {
        matches!(*self, ty::ReStatic)
    }

    #[inline]
    pub fn is_erased(self) -> bool {
        matches!(*self, ty::ReErased)
    }

    #[inline]
    pub fn is_bound(self) -> bool {
        matches!(*self, ty::ReBound(..))
    }

    #[inline]
    pub fn is_placeholder(self) -> bool {
        matches!(*self, ty::RePlaceholder(..))
    }

    #[inline]
    pub fn bound_at_or_above_binder(self, index: ty::DebruijnIndex) -> bool {
        match *self {
            ty::ReBound(debruijn, _) => debruijn >= index,
            _ => false,
        }
    }

    pub fn type_flags(self) -> TypeFlags {
        let mut flags = TypeFlags::empty();

        match *self {
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
            }
        }

        debug!("type_flags({:?}) = {:?}", self, flags);

        flags
    }

    /// Given an early-bound or free region, returns the `DefId` where it was bound.
    /// For example, consider the regions in this snippet of code:
    ///
    /// ```ignore (illustrative)
    /// impl<'a> Foo {
    /// //   ^^ -- early bound, declared on an impl
    ///
    ///     fn bar<'b, 'c>(x: &self, y: &'b u32, z: &'c u64) where 'static: 'c
    /// //         ^^  ^^     ^ anonymous, late-bound
    /// //         |   early-bound, appears in where-clauses
    /// //         late-bound, appears only in fn args
    ///     {..}
    /// }
    /// ```
    ///
    /// Here, `free_region_binding_scope('a)` would return the `DefId`
    /// of the impl, and for all the other highlighted regions, it
    /// would return the `DefId` of the function. In other cases (not shown), this
    /// function might return the `DefId` of a closure.
    pub fn free_region_binding_scope(self, tcx: TyCtxt<'_>) -> DefId {
        match *self {
            ty::ReEarlyParam(br) => tcx.parent(br.def_id),
            ty::ReLateParam(fr) => fr.scope,
            _ => bug!("free_region_binding_scope invoked on inappropriate region: {:?}", self),
        }
    }

    /// True for free regions other than `'static`.
    pub fn is_param(self) -> bool {
        matches!(*self, ty::ReEarlyParam(_) | ty::ReLateParam(_))
    }

    /// True for free region in the current context.
    ///
    /// This is the case for `'static` and param regions.
    pub fn is_free(self) -> bool {
        match *self {
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
}

impl<'tcx> Deref for Region<'tcx> {
    type Target = RegionKind<'tcx>;

    #[inline]
    fn deref(&self) -> &RegionKind<'tcx> {
        self.0.0
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, TyEncodable, TyDecodable, PartialOrd, Ord)]
#[derive(HashStable)]
pub struct EarlyParamRegion {
    pub def_id: DefId,
    pub index: u32,
    pub name: Symbol,
}

impl std::fmt::Debug for EarlyParamRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}, {}, {}", self.def_id, self.index, self.name)
    }
}

rustc_index::newtype_index! {
    /// A **region** (lifetime) **v**ariable **ID**.
    #[derive(HashStable)]
    #[encodable]
    #[orderable]
    #[debug_format = "'?{}"]
    pub struct RegionVid {}
}

impl Atom for RegionVid {
    fn index(self) -> usize {
        Idx::index(self)
    }
}

#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Hash, TyEncodable, TyDecodable, Copy)]
#[derive(HashStable)]
/// The parameter representation of late-bound function parameters, "some region
/// at least as big as the scope `fr.scope`".
pub struct LateParamRegion {
    pub scope: DefId,
    pub bound_region: BoundRegionKind,
}

#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Hash, TyEncodable, TyDecodable, Copy)]
#[derive(HashStable)]
pub enum BoundRegionKind {
    /// An anonymous region parameter for a given fn (&T)
    BrAnon,

    /// Named region parameters for functions (a in &'a T)
    ///
    /// The `DefId` is needed to distinguish free regions in
    /// the event of shadowing.
    BrNamed(DefId, Symbol),

    /// Anonymous region for the implicit env pointer parameter
    /// to a closure
    BrEnv,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, TyEncodable, TyDecodable, Debug, PartialOrd, Ord)]
#[derive(HashStable)]
pub struct BoundRegion {
    pub var: BoundVar,
    pub kind: BoundRegionKind,
}

impl BoundRegionKind {
    pub fn is_named(&self) -> bool {
        match *self {
            BoundRegionKind::BrNamed(_, name) => {
                name != kw::UnderscoreLifetime && name != kw::Empty
            }
            _ => false,
        }
    }

    pub fn get_name(&self) -> Option<Symbol> {
        if self.is_named() {
            match *self {
                BoundRegionKind::BrNamed(_, name) => return Some(name),
                _ => unreachable!(),
            }
        }

        None
    }

    pub fn get_id(&self) -> Option<DefId> {
        match *self {
            BoundRegionKind::BrNamed(id, _) => return Some(id),
            _ => None,
        }
    }
}
