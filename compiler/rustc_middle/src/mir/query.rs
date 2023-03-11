//! Values computed by queries that use MIR.

use crate::mir::{Body, ConstantKind, Promoted};
use crate::ty::{self, OpaqueHiddenType, Ty, TyCtxt};
use rustc_data_structures::unord::UnordSet;
use rustc_data_structures::vec_map::VecMap;
use rustc_errors::ErrorGuaranteed;
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_index::bit_set::BitMatrix;
use rustc_index::vec::{Idx, IndexVec};
use rustc_span::Span;
use rustc_target::abi::VariantIdx;
use smallvec::SmallVec;
use std::cell::Cell;
use std::fmt::{self, Debug};

use super::{Field, SourceInfo};

#[derive(Copy, Clone, PartialEq, TyEncodable, TyDecodable, HashStable, Debug)]
pub enum UnsafetyViolationKind {
    /// Unsafe operation outside `unsafe`.
    General,
    /// Unsafe operation in an `unsafe fn` but outside an `unsafe` block.
    /// Has to be handled as a lint for backwards compatibility.
    UnsafeFn,
}

#[derive(Copy, Clone, PartialEq, TyEncodable, TyDecodable, HashStable, Debug)]
pub enum UnsafetyViolationDetails {
    CallToUnsafeFunction,
    UseOfInlineAssembly,
    InitializingTypeWith,
    CastOfPointerToInt,
    UseOfMutableStatic,
    UseOfExternStatic,
    DerefOfRawPointer,
    AccessToUnionField,
    MutationOfLayoutConstrainedField,
    BorrowOfLayoutConstrainedField,
    CallToFunctionWith,
}

impl UnsafetyViolationDetails {
    pub fn description_and_note(&self) -> (&'static str, &'static str) {
        use UnsafetyViolationDetails::*;
        match self {
            CallToUnsafeFunction => (
                "call to unsafe function",
                "consult the function's documentation for information on how to avoid undefined \
                 behavior",
            ),
            UseOfInlineAssembly => (
                "use of inline assembly",
                "inline assembly is entirely unchecked and can cause undefined behavior",
            ),
            InitializingTypeWith => (
                "initializing type with `rustc_layout_scalar_valid_range` attr",
                "initializing a layout restricted type's field with a value outside the valid \
                 range is undefined behavior",
            ),
            CastOfPointerToInt => {
                ("cast of pointer to int", "casting pointers to integers in constants")
            }
            UseOfMutableStatic => (
                "use of mutable static",
                "mutable statics can be mutated by multiple threads: aliasing violations or data \
                 races will cause undefined behavior",
            ),
            UseOfExternStatic => (
                "use of extern static",
                "extern statics are not controlled by the Rust type system: invalid data, \
                 aliasing violations or data races will cause undefined behavior",
            ),
            DerefOfRawPointer => (
                "dereference of raw pointer",
                "raw pointers may be null, dangling or unaligned; they can violate aliasing rules \
                 and cause data races: all of these are undefined behavior",
            ),
            AccessToUnionField => (
                "access to union field",
                "the field may not be properly initialized: using uninitialized data will cause \
                 undefined behavior",
            ),
            MutationOfLayoutConstrainedField => (
                "mutation of layout constrained field",
                "mutating layout constrained fields cannot statically be checked for valid values",
            ),
            BorrowOfLayoutConstrainedField => (
                "borrow of layout constrained field with interior mutability",
                "references to fields of layout constrained fields lose the constraints. Coupled \
                 with interior mutability, the field can be changed to invalid values",
            ),
            CallToFunctionWith => (
                "call to function with `#[target_feature]`",
                "can only be called if the required target features are available",
            ),
        }
    }
}

#[derive(Copy, Clone, PartialEq, TyEncodable, TyDecodable, HashStable, Debug)]
pub struct UnsafetyViolation {
    pub source_info: SourceInfo,
    pub lint_root: hir::HirId,
    pub kind: UnsafetyViolationKind,
    pub details: UnsafetyViolationDetails,
}

#[derive(Copy, Clone, PartialEq, TyEncodable, TyDecodable, HashStable, Debug)]
pub enum UnusedUnsafe {
    /// `unsafe` block contains no unsafe operations
    /// > ``unnecessary `unsafe` block``
    Unused,
    /// `unsafe` block nested under another (used) `unsafe` block
    /// > ``… because it's nested under this `unsafe` block``
    InUnsafeBlock(hir::HirId),
}

#[derive(TyEncodable, TyDecodable, HashStable, Debug)]
pub struct UnsafetyCheckResult {
    /// Violations that are propagated *upwards* from this function.
    pub violations: Vec<UnsafetyViolation>,

    /// Used `unsafe` blocks in this function. This is used for the "unused_unsafe" lint.
    pub used_unsafe_blocks: UnordSet<hir::HirId>,

    /// This is `Some` iff the item is not a closure.
    pub unused_unsafes: Option<Vec<(hir::HirId, UnusedUnsafe)>>,
}

rustc_index::newtype_index! {
    #[derive(HashStable)]
    #[debug_format = "_{}"]
    pub struct GeneratorSavedLocal {}
}

#[derive(Clone, Debug, TyEncodable, TyDecodable, HashStable, TypeFoldable, TypeVisitable)]
pub struct GeneratorSavedTy<'tcx> {
    pub ty: Ty<'tcx>,
    /// Source info corresponding to the local in the original MIR body.
    pub source_info: SourceInfo,
    /// Whether the local should be ignored for trait bound computations.
    pub ignore_for_traits: bool,
}

/// The layout of generator state.
#[derive(Clone, TyEncodable, TyDecodable, HashStable, TypeFoldable, TypeVisitable)]
pub struct GeneratorLayout<'tcx> {
    /// The type of every local stored inside the generator.
    pub field_tys: IndexVec<GeneratorSavedLocal, GeneratorSavedTy<'tcx>>,

    /// Which of the above fields are in each variant. Note that one field may
    /// be stored in multiple variants.
    pub variant_fields: IndexVec<VariantIdx, IndexVec<Field, GeneratorSavedLocal>>,

    /// The source that led to each variant being created (usually, a yield or
    /// await).
    pub variant_source_info: IndexVec<VariantIdx, SourceInfo>,

    /// Which saved locals are storage-live at the same time. Locals that do not
    /// have conflicts with each other are allowed to overlap in the computed
    /// layout.
    #[type_foldable(identity)]
    #[type_visitable(ignore)]
    pub storage_conflicts: BitMatrix<GeneratorSavedLocal, GeneratorSavedLocal>,
}

impl Debug for GeneratorLayout<'_> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        /// Prints an iterator of (key, value) tuples as a map.
        struct MapPrinter<'a, K, V>(Cell<Option<Box<dyn Iterator<Item = (K, V)> + 'a>>>);
        impl<'a, K, V> MapPrinter<'a, K, V> {
            fn new(iter: impl Iterator<Item = (K, V)> + 'a) -> Self {
                Self(Cell::new(Some(Box::new(iter))))
            }
        }
        impl<'a, K: Debug, V: Debug> Debug for MapPrinter<'a, K, V> {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_map().entries(self.0.take().unwrap()).finish()
            }
        }

        /// Prints the generator variant name.
        struct GenVariantPrinter(VariantIdx);
        impl From<VariantIdx> for GenVariantPrinter {
            fn from(idx: VariantIdx) -> Self {
                GenVariantPrinter(idx)
            }
        }
        impl Debug for GenVariantPrinter {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                let variant_name = ty::GeneratorSubsts::variant_name(self.0);
                if fmt.alternate() {
                    write!(fmt, "{:9}({:?})", variant_name, self.0)
                } else {
                    write!(fmt, "{}", variant_name)
                }
            }
        }

        /// Forces its contents to print in regular mode instead of alternate mode.
        struct OneLinePrinter<T>(T);
        impl<T: Debug> Debug for OneLinePrinter<T> {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(fmt, "{:?}", self.0)
            }
        }

        fmt.debug_struct("GeneratorLayout")
            .field("field_tys", &MapPrinter::new(self.field_tys.iter_enumerated()))
            .field(
                "variant_fields",
                &MapPrinter::new(
                    self.variant_fields
                        .iter_enumerated()
                        .map(|(k, v)| (GenVariantPrinter(k), OneLinePrinter(v))),
                ),
            )
            .field("storage_conflicts", &self.storage_conflicts)
            .finish()
    }
}

#[derive(Debug, TyEncodable, TyDecodable, HashStable)]
pub struct BorrowCheckResult<'tcx> {
    /// All the opaque types that are restricted to concrete types
    /// by this function. Unlike the value in `TypeckResults`, this has
    /// unerased regions.
    pub concrete_opaque_types: VecMap<LocalDefId, OpaqueHiddenType<'tcx>>,
    pub closure_requirements: Option<ClosureRegionRequirements<'tcx>>,
    pub used_mut_upvars: SmallVec<[Field; 8]>,
    pub tainted_by_errors: Option<ErrorGuaranteed>,
}

/// The result of the `mir_const_qualif` query.
///
/// Each field (except `error_occurred`) corresponds to an implementer of the `Qualif` trait in
/// `rustc_const_eval/src/transform/check_consts/qualifs.rs`. See that file for more information on each
/// `Qualif`.
#[derive(Clone, Copy, Debug, Default, TyEncodable, TyDecodable, HashStable)]
pub struct ConstQualifs {
    pub has_mut_interior: bool,
    pub needs_drop: bool,
    pub needs_non_const_drop: bool,
    pub custom_eq: bool,
    pub tainted_by_errors: Option<ErrorGuaranteed>,
}

/// After we borrow check a closure, we are left with various
/// requirements that we have inferred between the free regions that
/// appear in the closure's signature or on its field types. These
/// requirements are then verified and proved by the closure's
/// creating function. This struct encodes those requirements.
///
/// The requirements are listed as being between various `RegionVid`. The 0th
/// region refers to `'static`; subsequent region vids refer to the free
/// regions that appear in the closure (or generator's) type, in order of
/// appearance. (This numbering is actually defined by the `UniversalRegions`
/// struct in the NLL region checker. See for example
/// `UniversalRegions::closure_mapping`.) Note the free regions in the
/// closure's signature and captures are erased.
///
/// Example: If type check produces a closure with the closure substs:
///
/// ```text
/// ClosureSubsts = [
///     'a,                                         // From the parent.
///     'b,
///     i8,                                         // the "closure kind"
///     for<'x> fn(&'<erased> &'x u32) -> &'x u32,  // the "closure signature"
///     &'<erased> String,                          // some upvar
/// ]
/// ```
///
/// We would "renumber" each free region to a unique vid, as follows:
///
/// ```text
/// ClosureSubsts = [
///     '1,                                         // From the parent.
///     '2,
///     i8,                                         // the "closure kind"
///     for<'x> fn(&'3 &'x u32) -> &'x u32,         // the "closure signature"
///     &'4 String,                                 // some upvar
/// ]
/// ```
///
/// Now the code might impose a requirement like `'1: '2`. When an
/// instance of the closure is created, the corresponding free regions
/// can be extracted from its type and constrained to have the given
/// outlives relationship.
#[derive(Clone, Debug, TyEncodable, TyDecodable, HashStable)]
pub struct ClosureRegionRequirements<'tcx> {
    /// The number of external regions defined on the closure. In our
    /// example above, it would be 3 -- one for `'static`, then `'1`
    /// and `'2`. This is just used for a sanity check later on, to
    /// make sure that the number of regions we see at the callsite
    /// matches.
    pub num_external_vids: usize,

    /// Requirements between the various free regions defined in
    /// indices.
    pub outlives_requirements: Vec<ClosureOutlivesRequirement<'tcx>>,
}

/// Indicates an outlives-constraint between a type or between two
/// free regions declared on the closure.
#[derive(Copy, Clone, Debug, TyEncodable, TyDecodable, HashStable)]
pub struct ClosureOutlivesRequirement<'tcx> {
    // This region or type ...
    pub subject: ClosureOutlivesSubject<'tcx>,

    // ... must outlive this one.
    pub outlived_free_region: ty::RegionVid,

    // If not, report an error here ...
    pub blame_span: Span,

    // ... due to this reason.
    pub category: ConstraintCategory<'tcx>,
}

// Make sure this enum doesn't unintentionally grow
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
rustc_data_structures::static_assert_size!(ConstraintCategory<'_>, 16);

/// Outlives-constraints can be categorized to determine whether and why they
/// are interesting (for error reporting). Order of variants indicates sort
/// order of the category, thereby influencing diagnostic output.
///
/// See also `rustc_const_eval::borrow_check::constraints`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord, Hash)]
#[derive(TyEncodable, TyDecodable, HashStable, Lift, TypeVisitable, TypeFoldable)]
pub enum ConstraintCategory<'tcx> {
    Return(ReturnConstraint),
    Yield,
    UseAsConst,
    UseAsStatic,
    TypeAnnotation,
    Cast,

    /// A constraint that came from checking the body of a closure.
    ///
    /// We try to get the category that the closure used when reporting this.
    ClosureBounds,

    /// Contains the function type if available.
    CallArgument(Option<Ty<'tcx>>),
    CopyBound,
    SizedBound,
    Assignment,
    /// A constraint that came from a usage of a variable (e.g. in an ADT expression
    /// like `Foo { field: my_val }`)
    Usage,
    OpaqueType,
    ClosureUpvar(Field),

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
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord, Hash)]
#[derive(TyEncodable, TyDecodable, HashStable, TypeVisitable, TypeFoldable)]
pub enum ReturnConstraint {
    Normal,
    ClosureUpvar(Field),
}

/// The subject of a `ClosureOutlivesRequirement` -- that is, the thing
/// that must outlive some region.
#[derive(Copy, Clone, Debug, TyEncodable, TyDecodable, HashStable)]
pub enum ClosureOutlivesSubject<'tcx> {
    /// Subject is a type, typically a type parameter, but could also
    /// be a projection. Indicates a requirement like `T: 'a` being
    /// passed to the caller, where the type here is `T`.
    Ty(ClosureOutlivesSubjectTy<'tcx>),

    /// Subject is a free region from the closure. Indicates a requirement
    /// like `'a: 'b` being passed to the caller; the region here is `'a`.
    Region(ty::RegionVid),
}

/// Represents a `ty::Ty` for use in [`ClosureOutlivesSubject`].
///
/// This abstraction is necessary because the type may include `ReVar` regions,
/// which is what we use internally within NLL code, and they can't be used in
/// a query response.
///
/// DO NOT implement `TypeVisitable` or `TypeFoldable` traits, because this
/// type is not recognized as a binder for late-bound region.
#[derive(Copy, Clone, Debug, TyEncodable, TyDecodable, HashStable)]
pub struct ClosureOutlivesSubjectTy<'tcx> {
    inner: Ty<'tcx>,
}

impl<'tcx> ClosureOutlivesSubjectTy<'tcx> {
    /// All regions of `ty` must be of kind `ReVar` and must represent
    /// universal regions *external* to the closure.
    pub fn bind(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Self {
        let inner = tcx.fold_regions(ty, |r, depth| match r.kind() {
            ty::ReVar(vid) => {
                let br = ty::BoundRegion {
                    var: ty::BoundVar::new(vid.index()),
                    kind: ty::BrAnon(vid.as_u32(), None),
                };
                tcx.mk_re_late_bound(depth, br)
            }
            _ => bug!("unexpected region in ClosureOutlivesSubjectTy: {r:?}"),
        });

        Self { inner }
    }

    pub fn instantiate(
        self,
        tcx: TyCtxt<'tcx>,
        mut map: impl FnMut(ty::RegionVid) -> ty::Region<'tcx>,
    ) -> Ty<'tcx> {
        tcx.fold_regions(self.inner, |r, depth| match r.kind() {
            ty::ReLateBound(debruijn, br) => {
                debug_assert_eq!(debruijn, depth);
                map(ty::RegionVid::new(br.var.index()))
            }
            _ => bug!("unexpected region {r:?}"),
        })
    }
}

/// The constituent parts of a mir constant of kind ADT or array.
#[derive(Copy, Clone, Debug, HashStable)]
pub struct DestructuredConstant<'tcx> {
    pub variant: Option<VariantIdx>,
    pub fields: &'tcx [ConstantKind<'tcx>],
}

/// Coverage information summarized from a MIR if instrumented for source code coverage (see
/// compiler option `-Cinstrument-coverage`). This information is generated by the
/// `InstrumentCoverage` MIR pass and can be retrieved via the `coverageinfo` query.
#[derive(Clone, TyEncodable, TyDecodable, Debug, HashStable)]
pub struct CoverageInfo {
    /// The total number of coverage region counters added to the MIR `Body`.
    pub num_counters: u32,

    /// The total number of coverage region counter expressions added to the MIR `Body`.
    pub num_expressions: u32,
}

/// Shims which make dealing with `WithOptConstParam` easier.
///
/// For more information on why this is needed, consider looking
/// at the docs for `WithOptConstParam` itself.
impl<'tcx> TyCtxt<'tcx> {
    #[inline]
    pub fn mir_const_qualif_opt_const_arg(
        self,
        def: ty::WithOptConstParam<LocalDefId>,
    ) -> ConstQualifs {
        if let Some(param_did) = def.const_param_did {
            self.mir_const_qualif_const_arg((def.did, param_did))
        } else {
            self.mir_const_qualif(def.did)
        }
    }

    #[inline]
    pub fn promoted_mir_opt_const_arg(
        self,
        def: ty::WithOptConstParam<DefId>,
    ) -> &'tcx IndexVec<Promoted, Body<'tcx>> {
        if let Some((did, param_did)) = def.as_const_arg() {
            self.promoted_mir_of_const_arg((did, param_did))
        } else {
            self.promoted_mir(def.did)
        }
    }

    #[inline]
    pub fn mir_for_ctfe_opt_const_arg(self, def: ty::WithOptConstParam<DefId>) -> &'tcx Body<'tcx> {
        if let Some((did, param_did)) = def.as_const_arg() {
            self.mir_for_ctfe_of_const_arg((did, param_did))
        } else {
            self.mir_for_ctfe(def.did)
        }
    }
}
