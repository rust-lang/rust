use std::fmt;

use derive_where::derive_where;
#[cfg(feature = "nightly")]
use rustc_macros::{Decodable_NoContext, Encodable_NoContext, StableHash_NoContext};
use rustc_type_ir_macros::{GenericTypeVisitable, Lift_Generic};
use tracing::debug;

use crate::inherent::*;
use crate::intern::Interned;
use crate::relate::{Relate, RelateResult, TypeRelation};
use crate::{
    BoundRegion, BoundRegionKind, BoundVar, BoundVarIndexKind, DebruijnIndex, FallibleTypeFolder,
    Flags, Interner, PlaceholderRegion, RegionKind, RegionVid, TypeFlags, TypeFoldable, TypeFolder,
    TypeVisitable, TypeVisitor,
};

/// Use this rather than `RegionKind`, whenever possible.
#[derive_where(Clone, Copy, PartialEq, Eq, Hash; I: Interner)]
#[cfg_attr(feature = "nightly", derive(StableHash_NoContext))]
#[cfg_attr(feature = "nightly", rustc_pass_by_value)]
#[derive(GenericTypeVisitable, Lift_Generic)]
pub struct Region<I: Interner>(pub I::InternedRegionKind);

// These are only the `inherent` trait methods that have been ported across
impl<I: Interner> Region<I> {
    #[inline]
    pub fn new_var(interner: I, v: RegionVid) -> Self {
        interner.intern_re_var(v)
    }

    pub fn get_name(self, interner: I) -> Option<I::Symbol> {
        match self.kind() {
            RegionKind::ReEarlyParam(ebr) => ebr.get_name(interner),
            RegionKind::ReBound(_, br) => br.kind.get_name(interner),
            RegionKind::ReLateParam(fr) => fr.kind.get_name(interner),
            RegionKind::ReStatic => Some(I::Symbol::KW_STATIC_LIFETIME),
            RegionKind::RePlaceholder(placeholder) => placeholder.bound.kind.get_name(interner),
            _ => None,
        }
    }

    pub fn get_name_or_anon(self, interner: I) -> I::Symbol {
        match self.get_name(interner) {
            Some(name) => name,
            None => I::Symbol::SYM_ANON,
        }
    }

    /// Given some item `binding_item`, check if this region is a generic parameter introduced by it
    /// or one of the parent generics. Returns the `DefId` of the parameter definition if so.
    pub fn opt_param_def_id(self, interner: I, binding_item: I::DefId) -> Option<I::DefId> {
        match self.kind() {
            RegionKind::ReEarlyParam(ebr) => {
                Some(interner.generics_of(binding_item).param_region_def_id(interner, ebr))
            }
            RegionKind::ReLateParam(param) => param.kind.get_def_id(),
            _ => None,
        }
    }

    /// Is this region named by the user?
    pub fn is_named(self, interner: I) -> bool {
        match self.kind() {
            RegionKind::ReEarlyParam(ebr) => ebr.is_named(interner),
            RegionKind::ReBound(_, br) => br.kind.is_named(interner),
            RegionKind::ReLateParam(fr) => fr.kind.is_named(interner),
            RegionKind::ReStatic => true,
            RegionKind::ReVar(..) => false,
            RegionKind::RePlaceholder(placeholder) => placeholder.bound.kind.is_named(interner),
            RegionKind::ReErased => false,
            RegionKind::ReError(_) => false,
        }
    }

    /// Constructs a `RegionKind::ReError` region and registers a delayed bug to ensure it gets
    /// used.
    #[track_caller]
    pub fn new_error_misc(interner: I) -> Self {
        Self::new_error_with_message(
            interner,
            I::Span::dummy(),
            "RegionKind::ReError constructed but no error reported",
        )
    }

    /// Constructs a `RegionKind::ReError` region and registers a delayed bug with the given `msg`
    /// to ensure it gets used.
    #[track_caller]
    pub fn new_error_with_message(interner: I, span: I::Span, msg: impl ToString) -> Self {
        let reported = interner.span_delayed_bug(span, msg);
        Self::new_error(interner, reported)
    }

    #[inline]
    pub fn new_late_param(interner: I, scope: I::DefId, kind: I::LateParamRegionKind) -> Self {
        interner.intern_region(RegionKind::ReLateParam(LateParamRegion { scope, kind }))
    }

    #[inline]
    pub fn new_early_param(interner: I, early_bound_region: I::EarlyParamRegion) -> Self {
        interner.intern_region(RegionKind::ReEarlyParam(early_bound_region))
    }

    /// Constructs a `RegionKind::ReError` region.
    #[track_caller]
    pub fn new_error(interner: I, guar: I::ErrorGuaranteed) -> Self {
        interner.intern_region(RegionKind::ReError(guar))
    }

    #[inline]
    pub fn new_bound(interner: I, debruijn: DebruijnIndex, bound_region: BoundRegion<I>) -> Self {
        interner.intern_bound_region(debruijn, bound_region)
    }

    #[inline]
    pub fn new_anon_bound(interner: I, debruijn: DebruijnIndex, var: BoundVar) -> Self {
        Self::new_bound(interner, debruijn, BoundRegion { var, kind: BoundRegionKind::Anon })
    }

    #[inline]
    pub fn new_canonical_bound(interner: I, var: BoundVar) -> Self {
        interner.intern_canonical_bound(var)
    }

    #[inline]
    pub fn new_placeholder(interner: I, placeholder: PlaceholderRegion<I>) -> Self {
        interner.intern_region(RegionKind::RePlaceholder(placeholder))
    }

    #[inline]
    pub fn new_static(interner: I) -> Self {
        interner.get_re_static_lifetime()
    }

    #[inline]
    pub fn is_bound(self) -> bool {
        matches!(self.0.get(), RegionKind::ReBound(..))
    }

    #[inline]
    pub fn is_error(self) -> bool {
        matches!(self.kind(), RegionKind::ReError(_))
    }

    #[inline]
    pub fn is_static(self) -> bool {
        matches!(self.kind(), RegionKind::ReStatic)
    }

    #[inline]
    pub fn is_erased(self) -> bool {
        matches!(self.kind(), RegionKind::ReErased)
    }

    #[inline]
    pub fn is_placeholder(self) -> bool {
        matches!(self.kind(), RegionKind::RePlaceholder(..))
    }

    /// True for free regions other than `'static`.
    pub fn is_param(self) -> bool {
        matches!(self.kind(), RegionKind::ReEarlyParam(_) | RegionKind::ReLateParam(_))
    }

    /// True for free region in the current context.
    ///
    /// This is the case for `'static` and param regions.
    pub fn is_free(self) -> bool {
        match self.kind() {
            RegionKind::ReStatic | RegionKind::ReEarlyParam(..) | RegionKind::ReLateParam(..) => {
                true
            }
            RegionKind::ReVar(..)
            | RegionKind::RePlaceholder(..)
            | RegionKind::ReBound(..)
            | RegionKind::ReErased
            | RegionKind::ReError(..) => false,
        }
    }

    pub fn is_var(self) -> bool {
        matches!(self.kind(), RegionKind::ReVar(_))
    }

    pub fn as_var(self) -> RegionVid {
        match self.kind() {
            RegionKind::ReVar(vid) => vid,
            _ => panic!("expected region {:?} to be of kind ReVar", self),
        }
    }

    // FIXME this should be made private and instead accessed via the
    // trait Flags
    #[inline]
    pub fn type_flags(self) -> TypeFlags {
        let mut flags = TypeFlags::empty();

        match self.0.get() {
            RegionKind::ReVar(..) => {
                flags = flags | TypeFlags::HAS_FREE_REGIONS;
                flags = flags | TypeFlags::HAS_FREE_LOCAL_REGIONS;
                flags = flags | TypeFlags::HAS_RE_INFER;
            }
            RegionKind::RePlaceholder(..) => {
                flags = flags | TypeFlags::HAS_FREE_REGIONS;
                flags = flags | TypeFlags::HAS_FREE_LOCAL_REGIONS;
                flags = flags | TypeFlags::HAS_RE_PLACEHOLDER;
            }
            RegionKind::ReEarlyParam(..) => {
                flags = flags | TypeFlags::HAS_FREE_REGIONS;
                flags = flags | TypeFlags::HAS_FREE_LOCAL_REGIONS;
                flags = flags | TypeFlags::HAS_RE_PARAM;
            }
            RegionKind::ReLateParam { .. } => {
                flags = flags | TypeFlags::HAS_FREE_REGIONS;
                flags = flags | TypeFlags::HAS_FREE_LOCAL_REGIONS;
            }
            RegionKind::ReStatic => {
                flags = flags | TypeFlags::HAS_FREE_REGIONS;
            }
            RegionKind::ReBound(BoundVarIndexKind::Canonical, _) => {
                flags = flags | TypeFlags::HAS_RE_BOUND;
                flags = flags | TypeFlags::HAS_CANONICAL_BOUND;
            }
            RegionKind::ReBound(BoundVarIndexKind::Bound(..), _) => {
                flags = flags | TypeFlags::HAS_RE_BOUND;
            }
            RegionKind::ReErased => {
                flags = flags | TypeFlags::HAS_RE_ERASED;
            }
            RegionKind::ReError(_) => {
                flags = flags | TypeFlags::HAS_FREE_REGIONS;
                flags = flags | TypeFlags::HAS_RE_ERROR;
            }
        }
        debug!("type_flags({:?}) = {:?}", self, flags);

        flags
    }

    #[inline]
    pub fn kind(self) -> RegionKind<I> {
        self.0.get()
    }

    #[inline]
    pub fn bound_at_or_above_binder(self, index: DebruijnIndex) -> bool {
        match self.kind() {
            RegionKind::ReBound(BoundVarIndexKind::Bound(debruijn), _) => debruijn >= index,
            _ => false,
        }
    }
}

impl<I: Interner> Flags for Region<I> {
    fn flags(&self) -> TypeFlags {
        self.type_flags()
    }

    fn outer_exclusive_binder(&self) -> DebruijnIndex {
        match self.kind() {
            RegionKind::ReBound(BoundVarIndexKind::Bound(debruijn), _) => debruijn.shifted_in(1),
            _ => crate::INNERMOST,
        }
    }
}

impl<I: Interner> IntoKind for Region<I> {
    type Kind = RegionKind<I>;

    fn kind(self) -> Self::Kind {
        self.0.get()
    }
}

impl<I: Interner> Relate<I> for Region<I> {
    fn relate<R: TypeRelation<I>>(
        relation: &mut R,
        a: Region<I>,
        b: Region<I>,
    ) -> RelateResult<I, Region<I>> {
        relation.regions(a, b)
    }
}

impl<I: Interner> fmt::Debug for Region<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.kind())
    }
}

impl<I: Interner> TypeVisitable<I> for Region<I> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result {
        visitor.visit_region(*self)
    }
}

impl<I: Interner> TypeFoldable<I> for Region<I> {
    fn try_fold_with<F: FallibleTypeFolder<I>>(self, folder: &mut F) -> Result<Self, F::Error> {
        folder.try_fold_region(self)
    }

    fn fold_with<F: TypeFolder<I>>(self, folder: &mut F) -> Self {
        folder.fold_region(self)
    }
}

#[derive_where(Clone, Copy, PartialEq, Eq, Hash; I: Interner)]
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, Decodable_NoContext, StableHash_NoContext)
)]
/// The parameter representation of late-bound function parameters, "some region
/// at least as big as the scope `fr.scope`".
///
/// Similar to a placeholder region as we create `LateParam` regions when entering a binder
/// except they are always in the root universe and instead of using a boundvar to distinguish
/// between others we use the `DefId` of the parameter. For this reason the `bound_region` field
/// should basically always be `BoundRegionKind::Named` as otherwise there is no way of telling
/// different parameters apart.
pub struct LateParamRegion<I: Interner> {
    pub scope: I::DefId,
    pub kind: I::LateParamRegionKind,
}

impl<I: Interner> fmt::Debug for LateParamRegion<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ReLateParam({:?}, {:?})", self.scope, self.kind)
    }
}
