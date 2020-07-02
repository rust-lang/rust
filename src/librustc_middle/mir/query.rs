//! Values computed by queries that use MIR.

use crate::ty::{self, Ty};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::Lrc;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_index::bit_set::BitMatrix;
use rustc_index::vec::IndexVec;
use rustc_span::{Span, Symbol};
use rustc_target::abi::VariantIdx;
use smallvec::SmallVec;
use std::cell::Cell;
use std::fmt::{self, Debug};

use super::{Field, SourceInfo};

#[derive(Copy, Clone, PartialEq, RustcEncodable, RustcDecodable, HashStable)]
pub enum UnsafetyViolationKind {
    /// Only permitted in regular `fn`s, prohibitted in `const fn`s.
    General,
    /// Permitted both in `const fn`s and regular `fn`s.
    GeneralAndConstFn,
    /// Borrow of packed field.
    /// Has to be handled as a lint for backwards compatibility.
    BorrowPacked,
    /// Unsafe operation in an `unsafe fn` but outside an `unsafe` block.
    /// Has to be handled as a lint for backwards compatibility.
    /// Should stay gated under `#![feature(unsafe_block_in_unsafe_fn)]`.
    UnsafeFn,
    /// Borrow of packed field in an `unsafe fn` but outside an `unsafe` block.
    /// Has to be handled as a lint for backwards compatibility.
    /// Should stay gated under `#![feature(unsafe_block_in_unsafe_fn)]`.
    UnsafeFnBorrowPacked,
}

#[derive(Copy, Clone, PartialEq, RustcEncodable, RustcDecodable, HashStable)]
pub struct UnsafetyViolation {
    pub source_info: SourceInfo,
    pub lint_root: hir::HirId,
    pub description: Symbol,
    pub details: Symbol,
    pub kind: UnsafetyViolationKind,
}

#[derive(Clone, RustcEncodable, RustcDecodable, HashStable)]
pub struct UnsafetyCheckResult {
    /// Violations that are propagated *upwards* from this function.
    pub violations: Lrc<[UnsafetyViolation]>,
    /// `unsafe` blocks in this function, along with whether they are used. This is
    /// used for the "unused_unsafe" lint.
    pub unsafe_blocks: Lrc<[(hir::HirId, bool)]>,
}

rustc_index::newtype_index! {
    pub struct GeneratorSavedLocal {
        derive [HashStable]
        DEBUG_FORMAT = "_{}",
    }
}

/// The layout of generator state.
#[derive(Clone, RustcEncodable, RustcDecodable, HashStable, TypeFoldable)]
pub struct GeneratorLayout<'tcx> {
    /// The type of every local stored inside the generator.
    pub field_tys: IndexVec<GeneratorSavedLocal, Ty<'tcx>>,

    /// Which of the above fields are in each variant. Note that one field may
    /// be stored in multiple variants.
    pub variant_fields: IndexVec<VariantIdx, IndexVec<Field, GeneratorSavedLocal>>,

    /// The source that led to each variant being created (usually, a yield or
    /// await).
    pub variant_source_info: IndexVec<VariantIdx, SourceInfo>,

    /// Which saved locals are storage-live at the same time. Locals that do not
    /// have conflicts with each other are allowed to overlap in the computed
    /// layout.
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

#[derive(Debug, RustcEncodable, RustcDecodable, HashStable)]
pub struct BorrowCheckResult<'tcx> {
    /// All the opaque types that are restricted to concrete types
    /// by this function. Unlike the value in `TypeckTables`, this has
    /// unerased regions.
    pub concrete_opaque_types: FxHashMap<DefId, ty::ResolvedOpaqueTy<'tcx>>,
    pub closure_requirements: Option<ClosureRegionRequirements<'tcx>>,
    pub used_mut_upvars: SmallVec<[Field; 8]>,
}

/// The result of the `mir_const_qualif` query.
///
/// Each field corresponds to an implementer of the `Qualif` trait in
/// `librustc_mir/transform/check_consts/qualifs.rs`. See that file for more information on each
/// `Qualif`.
#[derive(Clone, Copy, Debug, Default, RustcEncodable, RustcDecodable, HashStable)]
pub struct ConstQualifs {
    pub has_mut_interior: bool,
    pub needs_drop: bool,
    pub custom_eq: bool,
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
///
/// In some cases, we have to record outlives requirements between types and
/// regions as well. In that case, if those types include any regions, those
/// regions are recorded using their external names (`ReStatic`,
/// `ReEarlyBound`, `ReFree`). We use these because in a query response we
/// cannot use `ReVar` (which is what we use internally within the rest of the
/// NLL code).
#[derive(Clone, Debug, RustcEncodable, RustcDecodable, HashStable)]
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
#[derive(Copy, Clone, Debug, RustcEncodable, RustcDecodable, HashStable)]
pub struct ClosureOutlivesRequirement<'tcx> {
    // This region or type ...
    pub subject: ClosureOutlivesSubject<'tcx>,

    // ... must outlive this one.
    pub outlived_free_region: ty::RegionVid,

    // If not, report an error here ...
    pub blame_span: Span,

    // ... due to this reason.
    pub category: ConstraintCategory,
}

/// Outlives-constraints can be categorized to determine whether and why they
/// are interesting (for error reporting). Order of variants indicates sort
/// order of the category, thereby influencing diagnostic output.
///
/// See also `rustc_mir::borrow_check::constraints`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord, Hash)]
#[derive(RustcEncodable, RustcDecodable, HashStable)]
pub enum ConstraintCategory {
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
    CallArgument,
    CopyBound,
    SizedBound,
    Assignment,
    OpaqueType,
    ClosureUpvar(hir::HirId),

    /// A "boring" constraint (caused by the given location) is one that
    /// the user probably doesn't want to see described in diagnostics,
    /// because it is kind of an artifact of the type system setup.
    /// Example: `x = Foo { field: y }` technically creates
    /// intermediate regions representing the "type of `Foo { field: y
    /// }`", and data flows from `y` into those variables, but they
    /// are not very interesting. The assignment into `x` on the other
    /// hand might be.
    Boring,
    // Boring and applicable everywhere.
    BoringNoLocation,

    /// A constraint that doesn't correspond to anything the user sees.
    Internal,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord, Hash)]
#[derive(RustcEncodable, RustcDecodable, HashStable)]
pub enum ReturnConstraint {
    Normal,
    ClosureUpvar(hir::HirId),
}

/// The subject of a `ClosureOutlivesRequirement` -- that is, the thing
/// that must outlive some region.
#[derive(Copy, Clone, Debug, RustcEncodable, RustcDecodable, HashStable)]
pub enum ClosureOutlivesSubject<'tcx> {
    /// Subject is a type, typically a type parameter, but could also
    /// be a projection. Indicates a requirement like `T: 'a` being
    /// passed to the caller, where the type here is `T`.
    ///
    /// The type here is guaranteed not to contain any free regions at
    /// present.
    Ty(Ty<'tcx>),

    /// Subject is a free region from the closure. Indicates a requirement
    /// like `'a: 'b` being passed to the caller; the region here is `'a`.
    Region(ty::RegionVid),
}

/// The constituent parts of an ADT or array.
#[derive(Copy, Clone, Debug, HashStable)]
pub struct DestructuredConst<'tcx> {
    pub variant: Option<VariantIdx>,
    pub fields: &'tcx [&'tcx ty::Const<'tcx>],
}

/// Coverage information summarized from a MIR if instrumented for source code coverage (see
/// compiler option `-Zinstrument-coverage`). This information is generated by the
/// `InstrumentCoverage` MIR pass and can be retrieved via the `coverageinfo` query.
#[derive(Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct CoverageInfo {
    /// A hash value that can be used by the consumer of the coverage profile data to detect
    /// changes to the instrumented source of the associated MIR body (typically, for an
    /// individual function).
    pub hash: u64,

    /// The total number of coverage region counters added to the MIR `Body`.
    pub num_counters: u32,
}
