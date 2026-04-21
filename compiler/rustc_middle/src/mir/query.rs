//! Values computed by queries that use MIR.

use std::fmt::{self, Debug};

use rustc_abi::{FieldIdx, VariantIdx};
use rustc_data_structures::fx::FxIndexMap;
use rustc_data_structures::unord::UnordMap;
use rustc_errors::ErrorGuaranteed;
use rustc_hir::def_id::LocalDefId;
use rustc_index::IndexVec;
use rustc_index::bit_set::BitMatrix;
use rustc_macros::{HashStable, TyDecodable, TyEncodable, TypeFoldable, TypeVisitable};
use rustc_span::{Span, Symbol};

use super::{ConstValue, SourceInfo};
use crate::ty::{self, CoroutineArgsExt, Ty};

rustc_index::newtype_index! {
    #[stable_hash]
    #[encodable]
    #[debug_format = "_s{}"]
    pub struct CoroutineSavedLocal {}
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[derive(TyEncodable, TyDecodable, HashStable, TypeFoldable, TypeVisitable)]
pub struct CoroutineSavedTy<'tcx> {
    pub ty: Ty<'tcx>,
    /// Source info corresponding to the local in the original MIR body.
    pub source_info: SourceInfo,
    /// Whether the local should be ignored for trait bound computations.
    pub ignore_for_traits: bool,
}

/// The layout of coroutine state.
#[derive(Clone, PartialEq, Eq)]
#[derive(TyEncodable, TyDecodable, HashStable, TypeFoldable, TypeVisitable)]
pub struct CoroutineLayout<'tcx> {
    /// The type of every local stored inside the coroutine.
    pub field_tys: IndexVec<CoroutineSavedLocal, CoroutineSavedTy<'tcx>>,

    /// The name for debuginfo.
    pub field_names: IndexVec<CoroutineSavedLocal, Option<Symbol>>,

    /// Which of the above fields are in each variant. Note that one field may
    /// be stored in multiple variants.
    pub variant_fields: IndexVec<VariantIdx, IndexVec<FieldIdx, CoroutineSavedLocal>>,

    /// The source that led to each variant being created (usually, a yield or
    /// await).
    pub variant_source_info: IndexVec<VariantIdx, SourceInfo>,

    /// Which saved locals are storage-live at the same time. Locals that do not
    /// have conflicts with each other are allowed to overlap in the computed
    /// layout.
    #[type_foldable(identity)]
    #[type_visitable(ignore)]
    pub storage_conflicts: BitMatrix<CoroutineSavedLocal, CoroutineSavedLocal>,
}

impl Debug for CoroutineLayout<'_> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("CoroutineLayout")
            .field_with("field_tys", |fmt| {
                fmt.debug_map().entries(self.field_tys.iter_enumerated()).finish()
            })
            .field_with("variant_fields", |fmt| {
                let mut map = fmt.debug_map();
                for (idx, fields) in self.variant_fields.iter_enumerated() {
                    map.key_with(|fmt| {
                        let variant_name = ty::CoroutineArgs::variant_name(idx);
                        if fmt.alternate() {
                            write!(fmt, "{variant_name:9}({idx:?})")
                        } else {
                            write!(fmt, "{variant_name}")
                        }
                    });
                    // Force variant fields to print in regular mode instead of alternate mode.
                    map.value_with(|fmt| write!(fmt, "{fields:?}"));
                }
                map.finish()
            })
            .field("storage_conflicts", &self.storage_conflicts)
            .finish()
    }
}

/// The result of the `mir_const_qualif` query.
///
/// Each field (except `tainted_by_errors`) corresponds to an implementer of the `Qualif` trait in
/// `rustc_const_eval/src/transform/check_consts/qualifs.rs`. See that file for more information on each
/// `Qualif`.
#[derive(Clone, Copy, Debug, Default, TyEncodable, TyDecodable, HashStable)]
pub struct ConstQualifs {
    pub has_mut_interior: bool,
    pub needs_drop: bool,
    pub needs_non_const_drop: bool,
    pub tainted_by_errors: Option<ErrorGuaranteed>,
}
/// Outlives-constraints can be categorized to determine whether and why they
/// are interesting (for error reporting). Order of variants indicates sort
/// order of the category, thereby influencing diagnostic output.
///
/// See also `rustc_const_eval::borrow_check::constraints`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[derive(TyEncodable, TyDecodable, HashStable, TypeVisitable, TypeFoldable)]
pub enum ConstraintCategory<'tcx> {
    Return(ReturnConstraint),
    Yield,
    UseAsConst,
    UseAsStatic,
    TypeAnnotation(AnnotationSource),
    Cast {
        is_raw_ptr_dyn_type_cast: bool,
        /// Whether this cast is a coercion that was automatically inserted by the compiler.
        is_implicit_coercion: bool,
        /// Whether this is an unsizing coercion and if yes, this contains the target type.
        /// Region variables are erased to ReErased.
        unsize_to: Option<Ty<'tcx>>,
    },

    /// Contains the function type if available.
    CallArgument(Option<Ty<'tcx>>),
    CopyBound,
    SizedBound,
    Assignment,
    /// A constraint that came from a usage of a variable (e.g. in an ADT expression
    /// like `Foo { field: my_val }`)
    Usage,
    OpaqueType,
    ClosureUpvar(FieldIdx),

    /// A constraint from a user-written predicate
    /// with the provided span, written on the item
    /// with the given `DefId`
    Predicate(Span),

    /// A "boring" constraint (caused by the given location) is one that
    /// the user probably doesn't want to see described in diagnostics,
    /// because it is kind of an artifact of the type system setup.
    Boring,
    // Boring and applicable everywhere.
    BoringNoLocation,

    /// A constraint that doesn't correspond to anything the user sees.
    Internal,

    /// An internal constraint added when a region outlives a placeholder
    /// it cannot name and therefore has to outlive `'static`. The argument
    /// is the unnameable placeholder and the constraint is always between
    /// an SCC representative and `'static`.
    OutlivesUnnameablePlaceholder(
        #[type_foldable(identity)]
        #[type_visitable(ignore)]
        ty::RegionVid,
    ),
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[derive(TyEncodable, TyDecodable, HashStable, TypeVisitable, TypeFoldable)]
pub enum ReturnConstraint {
    Normal,
    ClosureUpvar(FieldIdx),
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[derive(TyEncodable, TyDecodable, HashStable, TypeVisitable, TypeFoldable)]
pub enum AnnotationSource {
    Ascription,
    Declaration,
    OpaqueCast,
    GenericArg,
}

/// The constituent parts of a mir constant of kind ADT or array.
#[derive(Copy, Clone, Debug, HashStable)]
pub struct DestructuredConstant<'tcx> {
    pub variant: Option<VariantIdx>,
    pub fields: &'tcx [(ConstValue, Ty<'tcx>)],
}

/// Projected outlives graph over a subset of SCCs.
/// Only SCCs containing regions involved in call-site mappings are
/// represented; intermediate SCCs are collapsed into direct edges.
#[derive(Clone, Debug, Default, TyEncodable, TyDecodable, HashStable)]
pub struct ProjectedOutlivesGraph {
    /// Maps RegionVid index to projected SCC index for each relevant region.
    /// Stored as an `UnordMap` for O(1) lookup by `RegionVid`.
    pub scc_of: UnordMap<u32, u32>,
    /// Adjacency list over projected SCCs: `scc_successors[i]` lists the
    /// projected SCC indices that SCC `i` outlives.
    pub scc_successors: Vec<Vec<u32>>,
}

impl ProjectedOutlivesGraph {
    /// Return the projected SCC for a RegionVid, if present.
    pub fn scc_of_vid(&self, vid: u32) -> Option<u32> {
        self.scc_of.get(&vid).copied()
    }

    /// Check whether `to` is a direct (1-hop) successor of `from` in the
    /// projected SCC graph. This captures immediate type-level outlives
    /// constraints without transitive closure, which is important for
    /// correctly distinguishing "derived from this param" vs "merely
    /// outlived by this param".
    pub fn scc_directly_reaches(&self, from: u32, to: u32) -> bool {
        if from == to {
            return true;
        }
        if let Some(succs) = self.scc_successors.get(from as usize) {
            succs.contains(&to)
        } else {
            false
        }
    }

    /// Check whether `from` can reach `to` in the SCC successors DAG
    /// (i.e., `from` outlives `to`). Uses BFS.
    pub fn scc_reaches(&self, from: u32, to: u32) -> bool {
        if from == to {
            return true;
        }
        let mut visited = rustc_data_structures::fx::FxHashSet::default();
        let mut stack = vec![from];
        while let Some(current) = stack.pop() {
            if current == to {
                return true;
            }
            if !visited.insert(current) {
                continue;
            }
            if let Some(succs) = self.scc_successors.get(current as usize) {
                stack.extend(succs.iter().copied());
            }
        }
        false
    }
}

/// Region mappings for a single call site.
#[derive(Clone, Debug, TyEncodable, TyDecodable, HashStable)]
pub struct CallSiteRegionMapping {
    /// Stable call-site identifier from the `call_id` field of
    /// `TerminatorKind::Call` / `TerminatorKind::TailCall`. This ID is
    /// assigned during MIR construction and is preserved across all MIR
    /// optimizations, allowing correlation between the borrowck region
    /// summary and the optimized MIR that the monomorphization collector
    /// walks.
    pub call_id: u32,
    /// Map from walk-order index to local RegionVid index.
    ///
    /// The walk-order index is assigned by a `TypeVisitor` depth-first walk
    /// over the callee's generic args, counting every region encountered
    /// in visitation order. Stored as an `UnordMap` for O(1) lookup by
    /// walk position.
    pub region_mappings: UnordMap<u32, u32>,
}

impl CallSiteRegionMapping {
    /// Return the local RegionVid stored at a given walk-order index.
    pub fn vid_for_walk_pos(&self, walk_pos: u32) -> Option<u32> {
        self.region_mappings.get(&walk_pos).copied()
    }
}

/// A structural slot within a monomorphized generic-arg list.
///
/// `arg_ordinal` identifies the outer generic argument in the list and
/// `offset_within_arg` identifies the region slot within that argument's
/// TypeVisitor DFS walk.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[derive(TyEncodable, TyDecodable, HashStable)]
pub struct InputSlot {
    pub arg_ordinal: u32,
    pub offset_within_arg: u32,
}

/// Provenance for a RegionVid in a borrowck region summary.
///
/// `Input` means the vid originates from one of the body's instantiated
/// input slots. `LocalOnly` means the vid is body-local and should not be
/// transported across body boundaries.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[derive(TyEncodable, TyDecodable, HashStable)]
pub enum VidProvenance {
    Static,
    Input(InputSlot),
    /// The vid is bounded from above by a single non-static universal
    /// region (the unsizing edge's "lifetime GCD"). Post-borrowck, the
    /// concrete lifetime through the coercion is this universal, even
    /// though the NLL constraint graph only records the forward direction
    /// (`'universal: vid`) due to dyn-type covariance.
    BoundedByUniversal(InputSlot),
    LocalOnly,
}

/// Sentinel param position representing `'static` in
/// [`BorrowckRegionSummary::vid_to_param_pos`].
pub const STATIC_PARAM_POS: u32 = u32::MAX;

/// Compact summary of borrowck's solved region constraints.
/// Provides cross-crate access to outlives relationships that
/// only exist transiently during borrowck's region inference.
#[derive(Clone, Debug, Default, TyEncodable, TyDecodable, HashStable)]
pub struct BorrowckRegionSummary {
    /// Call-site region mappings: for each call site in the MIR body,
    /// records which local RegionVid instantiates each of the callee's
    /// generic lifetime parameters.
    pub call_site_mappings: UnordMap<u32, CallSiteRegionMapping>,
    /// Projected outlives graph over the SCCs of regions involved in
    /// call-site mappings.
    pub outlives_graph: ProjectedOutlivesGraph,
    /// Per-vid provenance for relevant regions consumed by the current
    /// body. Used by call-chain composition to tell whether a region is
    /// input-sourced or genuinely local.
    pub vid_provenance: UnordMap<u32, VidProvenance>,
    /// Maps each universal RegionVid to its param position.
    ///
    /// For `ReEarlyParam`: the param's `index`.
    /// For named `ReLateParam`: the RegionVid itself (identity).
    /// For `ReStatic`: [`STATIC_PARAM_POS`].
    ///
    /// Consumers that need the reverse mapping (param position → RegionVid)
    /// can invert this cheaply — it is a small vec (one entry per universal
    /// region).
    pub vid_to_param_pos: Vec<(u32, u32)>,
}

/// Combined borrowck result containing both hidden types (for opaque type
/// inference) and region summaries (for cross-function outlives propagation).
///
/// This is the shared core computation that both `mir_borrowck` and
/// `borrowck_region_summary` delegate to.
#[derive(Debug, HashStable)]
pub struct BorrowckResult<'tcx> {
    pub hidden_types:
        Result<FxIndexMap<LocalDefId, ty::DefinitionSiteHiddenType<'tcx>>, ErrorGuaranteed>,
    pub region_summaries: FxIndexMap<LocalDefId, BorrowckRegionSummary>,
}
