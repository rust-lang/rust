pub mod canonical;

use crate::ty::Region;
use crate::ty::Ty;
use rustc_data_structures::sync::Lrc;
use rustc_hir::def_id::DefId;
use rustc_span::Span;

/// Requires that `region` must be equal to one of the regions in `choice_regions`.
/// We often denote this using the syntax:
///
/// ```
/// R0 member of [O1..On]
/// ```
#[derive(Debug, Clone, HashStable, TypeFoldable, Lift)]
pub struct MemberConstraint<'tcx> {
    /// The `DefId` of the opaque type causing this constraint: used for error reporting.
    pub opaque_type_def_id: DefId,

    /// The span where the hidden type was instantiated.
    pub definition_span: Span,

    /// The hidden type in which `member_region` appears: used for error reporting.
    pub hidden_ty: Ty<'tcx>,

    /// The region `R0`.
    pub member_region: Region<'tcx>,

    /// The options `O1..On`.
    pub choice_regions: Lrc<Vec<Region<'tcx>>>,
}
