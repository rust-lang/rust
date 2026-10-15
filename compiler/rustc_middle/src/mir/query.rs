//! Values computed by queries that use MIR.

use std::fmt::{self, Debug};

use rustc_abi::{FieldIdx, VariantIdx};
use rustc_data_structures::fx::FxIndexMap;
use rustc_errors::ErrorGuaranteed;
use rustc_index::IndexVec;
use rustc_index::bit_set::{BitMatrix, DenseBitSet};
use rustc_macros::{StableHash, TyDecodable, TyEncodable, TypeFoldable, TypeVisitable};
use rustc_span::def_id::LocalDefId;
use rustc_span::{Span, Symbol};

use super::{ConstValue, Mutability, Place, SourceInfo, StatementId};
use crate::ty::{self, CoroutineArgsExt, Ty};

rustc_index::newtype_index! {
    #[stable_hash]
    #[encodable]
    #[debug_format = "_s{}"]
    pub struct CoroutineSavedLocal {}
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[derive(TyEncodable, TyDecodable, StableHash, TypeFoldable, TypeVisitable)]
pub struct CoroutineSavedTy<'tcx> {
    pub ty: Ty<'tcx>,
    /// Source info corresponding to the local in the original MIR body.
    pub source_info: SourceInfo,
    /// Whether the local should be ignored for trait bound computations.
    pub ignore_for_traits: bool,
    /// The name for debuginfo.
    pub debuginfo_name: Option<Symbol>,
}

/// The layout of coroutine state.
#[derive(Clone, PartialEq, Eq)]
#[derive(TyEncodable, TyDecodable, StableHash, TypeFoldable, TypeVisitable)]
pub struct CoroutineLayout<'tcx> {
    /// The type of every local stored inside the coroutine.
    pub field_tys: IndexVec<CoroutineSavedLocal, CoroutineSavedTy<'tcx>>,

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
#[derive(Clone, Copy, Debug, Default, TyEncodable, TyDecodable, StableHash)]
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
#[derive(TyEncodable, TyDecodable, StableHash, TypeVisitable, TypeFoldable)]
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

    // FIXME(-Zassumptions-on-binders): this is a temporary hack until we support
    // proper diagnostics for solver region constraints.
    SolverRegionConstraint(Span),
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[derive(TyEncodable, TyDecodable, StableHash, TypeVisitable, TypeFoldable)]
pub enum ReturnConstraint {
    Normal,
    ClosureUpvar(FieldIdx),
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[derive(TyEncodable, TyDecodable, StableHash, TypeVisitable, TypeFoldable)]
pub enum AnnotationSource {
    Ascription,
    Declaration,
    OpaqueCast,
    GenericArg,
}

/// The constituent parts of a mir constant of kind ADT or array.
#[derive(Copy, Clone, Debug, StableHash)]
pub struct DestructuredConstant<'tcx> {
    pub variant: Option<VariantIdx>,
    pub fields: &'tcx [(ConstValue, Ty<'tcx>)],
}

/// Saves the borrow checker results to generate the set of writes we need to checkpoint
#[derive(Debug, Default, StableHash)]
pub struct BorrowCheckResult<'tcx> {
    /// The fully resolved hidden types of the opaque types the bodies define
    pub hidden_types: FxIndexMap<LocalDefId, ty::DefinitionSiteHiddenType<'tcx>>,
    /// Also saves the Polonius type-checker liveness stuff, per body
    pub loans: FxIndexMap<LocalDefId, BodyLoans<'tcx>>,
}

rustc_index::newtype_index! {
    /// A loan in [`BodyLoans::loans`]
    #[stable_hash]
    #[encodable]
    #[orderable]
    #[debug_format = "l{}"]
    pub struct LoanId {}
}

/// The write set of the two checkpoints at each `.await` of a coroutine.
#[derive(Clone, Debug, TyEncodable, TyDecodable, StableHash)]
pub struct CheckpointTable {
    /// One entry per `.await`, in block order of their `Yield`s, which is also the order of
    /// `StateTransform`'s suspend states (state `3 + index`)
    pub awaits: Vec<AwaitCheckpoints>,
}

/// The two checkpoints of one `.await`
#[derive(Clone, Debug, TyEncodable, TyDecodable, StableHash)]
pub struct AwaitCheckpoints {
    /// Values written to from the last `.await`'s completion to the start of the current `.await`
    pub before: Vec<(CheckpointObject, Depth)>,
    /// The side effects of the `.await` call
    pub effects: Vec<(CheckpointObject, Depth)>,
}

TrivialTypeTraversalImpls! { CheckpointTable }

/// Something a checkpoint copies
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, TyEncodable, TyDecodable, StableHash)]
pub enum CheckpointObject {
    /// The coroutine's state, which `StateTransform` writes at every suspension
    Discriminant,
    /// A local `StateTransform` stores in the future, as slot `_sN` of the coroutine layout
    Saved(CoroutineSavedLocal),
    /// A captured upvar (an argument of the async fn), or all of them if the field is unknown
    Upvar(Option<FieldIdx>),
    /// Memory outside the future, reached through a reference
    Outside,
}

/// How much of an object a checkpoint copies
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, TyEncodable, TyDecodable, StableHash)]
pub enum Depth {
    /// Only the object's own bytes
    Shallow,
    /// The object and everything reachable through the references inside it
    Deep,
}

/// A list of loans for each body and the liveness during each statement
#[derive(Debug, StableHash)]
pub struct BodyLoans<'tcx> {
    /// Alias analysis results from Polonius mapped to each LoanId
    pub loans: IndexVec<LoanId, Option<(Place<'tcx>, Mutability)>>,
    /// Instructions that need an alias analysis and the liveness info
    pub instructions: FxIndexMap<StatementId, InstructionLoans>,
}

/// Instructions that are writes through a pointer, calls, drops or suspensions
#[derive(Debug, StableHash)]
pub struct InstructionLoans {
    /// Live loans at the instruction's point
    pub live: DenseBitSet<LoanId>,
    /// Set of eligible loans that are still alive at the instruction
    pub reachable: Option<DenseBitSet<LoanId>>,
    /// Whether those pointers can also hold memory not borrowed in this body
    pub outside: bool,
}
