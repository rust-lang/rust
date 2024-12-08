pub mod canonical;
pub mod unify_key;

use rustc_data_structures::sync::Lrc;
use rustc_macros::{HashStable, TypeFoldable, TypeVisitable};
use rustc_span::Span;

use crate::ty::{OpaqueTypeKey, Region, Ty};

/// Requires that `region` must be equal to one of the regions in `choice_regions`.
/// We often denote this using the syntax:
///
/// ```text
/// R0 member of [O1..On]
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[derive(HashStable, TypeFoldable, TypeVisitable)]
pub struct MemberConstraint<'tcx> {
    /// The `DefId` and args of the opaque type causing this constraint.
    /// Used for error reporting.
    pub key: OpaqueTypeKey<'tcx>,

    /// The span where the hidden type was instantiated.
    pub definition_span: Span,

    /// The hidden type in which `member_region` appears: used for error reporting.
    pub hidden_ty: Ty<'tcx>,

    /// The region `R0`.
    pub member_region: Region<'tcx>,

    /// The options `O1..On`.
    pub choice_regions: Lrc<Vec<Region<'tcx>>>,
}
