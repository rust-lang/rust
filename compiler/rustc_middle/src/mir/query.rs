//! Values computed by queries that use MIR.

use std::fmt::{self, Debug};

use rustc_abi::{FieldIdx, VariantIdx};
use rustc_data_structures::fx::FxIndexMap;
use rustc_errors::ErrorGuaranteed;
use rustc_hir::def_id::LocalDefId;
use rustc_index::IndexVec;
use rustc_index::bit_set::BitMatrix;
use rustc_macros::{HashStable, TyDecodable, TyEncodable, TypeFoldable, TypeVisitable};
use rustc_span::{Span, Symbol};

use super::{ConstValue, SourceInfo};
use crate::ty::{self, CoroutineArgsExt, OpaqueHiddenType, Ty};

rustc_index::newtype_index! {
    #[derive(HashStable)]
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

/// All the opaque types that have had their hidden type fully computed.
/// Unlike the value in `TypeckResults`, this has unerased regions.
#[derive(Default, Debug, TyEncodable, TyDecodable, HashStable)]
pub struct DefinitionSiteHiddenTypes<'tcx>(pub FxIndexMap<LocalDefId, OpaqueHiddenType<'tcx>>);

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
