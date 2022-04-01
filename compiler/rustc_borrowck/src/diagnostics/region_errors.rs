//! Error reporting machinery for lifetime errors.

use rustc_middle::mir::ConstraintCategory;
use rustc_middle::ty::{self, RegionVid};

use crate::{nll::ConstraintDescription, MirBorrowckCtxt};

impl ConstraintDescription for ConstraintCategory {
    fn description(&self) -> &'static str {
        // Must end with a space. Allows for empty names to be provided.
        match self {
            ConstraintCategory::Assignment => "assignment ",
            ConstraintCategory::Return(_) => "returning this value ",
            ConstraintCategory::Yield => "yielding this value ",
            ConstraintCategory::UseAsConst => "using this value as a constant ",
            ConstraintCategory::UseAsStatic => "using this value as a static ",
            ConstraintCategory::Cast => "cast ",
            ConstraintCategory::CallArgument => "argument ",
            ConstraintCategory::TypeAnnotation => "type annotation ",
            ConstraintCategory::ClosureBounds => "closure body ",
            ConstraintCategory::SizedBound => "proving this value is `Sized` ",
            ConstraintCategory::CopyBound => "copying this value ",
            ConstraintCategory::OpaqueType => "opaque type ",
            ConstraintCategory::ClosureUpvar(_) => "closure capture ",
            ConstraintCategory::Usage => "this usage ",
            ConstraintCategory::Predicate(_)
            | ConstraintCategory::Boring
            | ConstraintCategory::BoringNoLocation
            | ConstraintCategory::Internal => "",
        }
    }
}

/// A collection of errors encountered during region inference. This is needed to efficiently
/// report errors after borrow checking.
///
/// Usually we expect this to either be empty or contain a small number of items, so we can avoid
/// allocation most of the time.
crate type RegionErrors = Vec<RegionErrorKind>;

#[derive(Clone, Debug)]
crate enum RegionErrorKind {
    /// An unexpected hidden region for an opaque type.
    UnexpectedHiddenRegion,

    /// Higher-ranked subtyping error.
    BoundUniversalRegionError,

    /// Any other lifetime error.
    RegionError,
}

/// Information about the various region constraints involved in a borrow checker error.
#[derive(Clone, Debug)]
pub struct ErrorConstraintInfo {}

impl<'a, 'tcx> MirBorrowckCtxt<'a, 'tcx> {
    /// Converts a region inference variable into a `ty::Region` that
    /// we can use for error reporting. If `r` is universally bound,
    /// then we use the name that we have on record for it. If `r` is
    /// existentially bound, then we check its inferred value and try
    /// to find a good name from that. Returns `None` if we can't find
    /// one (e.g., this is just some random part of the CFG).
    pub(super) fn to_error_region(&self, r: RegionVid) -> Option<ty::Region<'tcx>> {
        self.to_error_region_vid(r).and_then(|r| self.regioncx.region_definition(r).external_name)
    }

    /// Returns the `RegionVid` corresponding to the region returned by
    /// `to_error_region`.
    pub(super) fn to_error_region_vid(&self, r: RegionVid) -> Option<RegionVid> {
        if self.regioncx.universal_regions().is_universal_region(r) {
            Some(r)
        } else {
            // We just want something nameable, even if it's not
            // actually an upper bound.
            let upper_bound = self.regioncx.approx_universal_upper_bound(r);

            if self.regioncx.upper_bound_in_region_scc(r, upper_bound) {
                self.to_error_region_vid(upper_bound)
            } else {
                None
            }
        }
    }
}
