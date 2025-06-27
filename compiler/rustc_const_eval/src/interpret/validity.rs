//! Check the validity invariant of a given value, and tell the user
//! where in the value it got violated.
//! In const context, this goes even further and tries to approximate const safety.
//! That's useful because it means other passes (e.g. promotion) can rely on `const`s
//! to be const-safe.

use std::borrow::Cow;
use std::fmt::Write;
use std::hash::Hash;
use std::num::NonZero;

use either::{Left, Right};
use hir::def::DefKind;
use rustc_abi::{
    BackendRepr, FieldIdx, FieldsShape, Scalar as ScalarAbi, Size, VariantIdx, Variants,
    WrappingRange,
};
use rustc_ast::Mutability;
use rustc_data_structures::fx::FxHashSet;
use rustc_hir as hir;
use rustc_middle::bug;
use rustc_middle::mir::interpret::ValidationErrorKind::{self, *};
use rustc_middle::mir::interpret::{
    ExpectedKind, InterpErrorKind, InvalidMetaKind, Misalignment, PointerKind, Provenance,
    UnsupportedOpInfo, ValidationErrorInfo, alloc_range, interp_ok,
};
use rustc_middle::ty::layout::{LayoutCx, TyAndLayout};
use rustc_middle::ty::{self, Ty};
use rustc_span::{Symbol, sym};
use tracing::trace;

use super::machine::AllocMap;
use super::{
    AllocId, CheckInAllocMsg, GlobalAlloc, ImmTy, Immediate, InterpCx, InterpResult, MPlaceTy,
    Machine, MemPlaceMeta, PlaceTy, Pointer, Projectable, Scalar, ValueVisitor, err_ub,
    format_interp_error,
};
use crate::enter_trace_span;

// for the validation errors
#[rustfmt::skip]
use super::InterpErrorKind::UndefinedBehavior as Ub;
use super::InterpErrorKind::Unsupported as Unsup;
use super::UndefinedBehaviorInfo::*;
use super::UnsupportedOpInfo::*;

macro_rules! err_validation_failure {
    ($where:expr, $kind: expr) => {{
        let where_ = &$where;
        let path = if !where_.is_empty() {
            let mut path = String::new();
            write_path(&mut path, where_);
            Some(path)
        } else {
            None
        };

        err_ub!(ValidationError(ValidationErrorInfo { path, kind: $kind }))
    }};
}

macro_rules! throw_validation_failure {
    ($where:expr, $kind: expr) => {
        do yeet err_validation_failure!($where, $kind)
    };
}

/// If $e throws an error matching the pattern, throw a validation failure.
/// Other errors are passed back to the caller, unchanged -- and if they reach the root of
/// the visitor, we make sure only validation errors and `InvalidProgram` errors are left.
/// This lets you use the patterns as a kind of validation list, asserting which errors
/// can possibly happen:
///
/// ```ignore(illustrative)
/// let v = try_validation!(some_fn(), some_path, {
///     Foo | Bar | Baz => { "some failure" },
/// });
/// ```
///
/// The patterns must be of type `UndefinedBehaviorInfo`.
/// An additional expected parameter can also be added to the failure message:
///
/// ```ignore(illustrative)
/// let v = try_validation!(some_fn(), some_path, {
///     Foo | Bar | Baz => { "some failure" } expected { "something that wasn't a failure" },
/// });
/// ```
///
/// An additional nicety is that both parameters actually take format args, so you can just write
/// the format string in directly:
///
/// ```ignore(illustrative)
/// let v = try_validation!(some_fn(), some_path, {
///     Foo | Bar | Baz => { "{:?}", some_failure } expected { "{}", expected_value },
/// });
/// ```
///
macro_rules! try_validation {
    ($e:expr, $where:expr,
    $( $( $p:pat_param )|+ => $kind: expr ),+ $(,)?
    ) => {{
        $e.map_err_kind(|e| {
            // We catch the error and turn it into a validation failure. We are okay with
            // allocation here as this can only slow down builds that fail anyway.
            match e {
                $(
                    $($p)|+ => {
                        err_validation_failure!(
                            $where,
                            $kind
                        )
                    }
                ),+,
                e => e,
            }
        })?
    }};
}

/// We want to show a nice path to the invalid field for diagnostics,
/// but avoid string operations in the happy case where no error happens.
/// So we track a `Vec<PathElem>` where `PathElem` contains all the data we
/// need to later print something for the user.
#[derive(Copy, Clone, Debug)]
pub enum PathElem {
    Field(Symbol),
    Variant(Symbol),
    CoroutineState(VariantIdx),
    CapturedVar(Symbol),
    ArrayElem(usize),
    TupleElem(usize),
    Deref,
    EnumTag,
    CoroutineTag,
    DynDowncast,
    Vtable,
}

/// Extra things to check for during validation of CTFE results.
#[derive(Copy, Clone)]
pub enum CtfeValidationMode {
    /// Validation of a `static`
    Static { mutbl: Mutability },
    /// Validation of a promoted.
    Promoted,
    /// Validation of a `const`.
    /// `allow_immutable_unsafe_cell` says whether we allow `UnsafeCell` in immutable memory (which is the
    /// case for the top-level allocation of a `const`, where this is fine because the allocation will be
    /// copied at each use site).
    Const { allow_immutable_unsafe_cell: bool },
}

impl CtfeValidationMode {
    fn allow_immutable_unsafe_cell(self) -> bool {
        match self {
            CtfeValidationMode::Static { .. } => false,
            CtfeValidationMode::Promoted { .. } => false,
            CtfeValidationMode::Const { allow_immutable_unsafe_cell, .. } => {
                allow_immutable_unsafe_cell
            }
        }
    }
}

/// State for tracking recursive validation of references
pub struct RefTracking<T, PATH = ()> {
    seen: FxHashSet<T>,
    todo: Vec<(T, PATH)>,
}

impl<T: Clone + Eq + Hash + std::fmt::Debug, PATH: Default> RefTracking<T, PATH> {
    pub fn empty() -> Self {
        RefTracking { seen: FxHashSet::default(), todo: vec![] }
    }
    pub fn new(val: T) -> Self {
        let mut ref_tracking_for_consts =
            RefTracking { seen: FxHashSet::default(), todo: vec![(val.clone(), PATH::default())] };
        ref_tracking_for_consts.seen.insert(val);
        ref_tracking_for_consts
    }
    pub fn next(&mut self) -> Option<(T, PATH)> {
        self.todo.pop()
    }

    fn track(&mut self, val: T, path: impl FnOnce() -> PATH) {
        if self.seen.insert(val.clone()) {
            trace!("Recursing below ptr {:#?}", val);
            let path = path();
            // Remember to come back to this later.
            self.todo.push((val, path));
        }
    }
}

// FIXME make this translatable as well?
/// Format a path
fn write_path(out: &mut String, path: &[PathElem]) {
    use self::PathElem::*;

    for elem in path.iter() {
        match elem {
            Field(name) => write!(out, ".{name}"),
            EnumTag => write!(out, ".<enum-tag>"),
            Variant(name) => write!(out, ".<enum-variant({name})>"),
            CoroutineTag => write!(out, ".<coroutine-tag>"),
            CoroutineState(idx) => write!(out, ".<coroutine-state({})>", idx.index()),
            CapturedVar(name) => write!(out, ".<captured-var({name})>"),
            TupleElem(idx) => write!(out, ".{idx}"),
            ArrayElem(idx) => write!(out, "[{idx}]"),
            // `.<deref>` does not match Rust syntax, but it is more readable for long paths -- and
            // some of the other items here also are not Rust syntax. Actually we can't
            // even use the usual syntax because we are just showing the projections,
            // not the root.
            Deref => write!(out, ".<deref>"),
            DynDowncast => write!(out, ".<dyn-downcast>"),
            Vtable => write!(out, ".<vtable>"),
        }
        .unwrap()
    }
}

/// Represents a set of `Size` values as a sorted list of ranges.
// These are (offset, length) pairs, and they are sorted and mutually disjoint,
// and never adjacent (i.e. there's always a gap between two of them).
#[derive(Debug, Clone)]
pub struct RangeSet(Vec<(Size, Size)>);

impl RangeSet {
    fn add_range(&mut self, offset: Size, size: Size) {
        if size.bytes() == 0 {
            // No need to track empty ranges.
            return;
        }
        let v = &mut self.0;
        // We scan for a partition point where the left partition is all the elements that end
        // strictly before we start. Those are elements that are too "low" to merge with us.
        let idx =
            v.partition_point(|&(other_offset, other_size)| other_offset + other_size < offset);
        // Now we want to either merge with the first element of the second partition, or insert ourselves before that.
        if let Some(&(other_offset, other_size)) = v.get(idx)
            && offset + size >= other_offset
        {
            // Their end is >= our start (otherwise it would not be in the 2nd partition) and
            // our end is >= their start. This means we can merge the ranges.
            let new_start = other_offset.min(offset);
            let mut new_end = (other_offset + other_size).max(offset + size);
            // We grew to the right, so merge with overlapping/adjacent elements.
            // (We also may have grown to the left, but that can never make us adjacent with
            // anything there since we selected the first such candidate via `partition_point`.)
            let mut scan_right = 1;
            while let Some(&(next_offset, next_size)) = v.get(idx + scan_right)
                && new_end >= next_offset
            {
                // Increase our size to absorb the next element.
                new_end = new_end.max(next_offset + next_size);
                // Look at the next element.
                scan_right += 1;
            }
            // Update the element we grew.
            v[idx] = (new_start, new_end - new_start);
            // Remove the elements we absorbed (if any).
            if scan_right > 1 {
                drop(v.drain((idx + 1)..(idx + scan_right)));
            }
        } else {
            // Insert new element.
            v.insert(idx, (offset, size));
        }
    }
}

struct ValidityVisitor<'rt, 'tcx, M: Machine<'tcx>> {
    /// The `path` may be pushed to, but the part that is present when a function
    /// starts must not be changed!  `visit_fields` and `visit_array` rely on
    /// this stack discipline.
    path: Vec<PathElem>,
    ref_tracking: Option<&'rt mut RefTracking<MPlaceTy<'tcx, M::Provenance>, Vec<PathElem>>>,
    /// `None` indicates this is not validating for CTFE (but for runtime).
    ctfe_mode: Option<CtfeValidationMode>,
    ecx: &'rt mut InterpCx<'tcx, M>,
    /// Whether provenance should be reset outside of pointers (emulating the effect of a typed
    /// copy).
    reset_provenance_and_padding: bool,
    /// This tracks which byte ranges in this value contain data; the remaining bytes are padding.
    /// The ideal representation here would be pointer-length pairs, but to keep things more compact
    /// we only store a (range) set of offsets -- the base pointer is the same throughout the entire
    /// visit, after all.
    /// If this is `Some`, then `reset_provenance_and_padding` must be true (but not vice versa:
    /// we might not track data vs padding bytes if the operand isn't stored in memory anyway).
    data_bytes: Option<RangeSet>,
}

impl<'rt, 'tcx, M: Machine<'tcx>> ValidityVisitor<'rt, 'tcx, M> {
    fn aggregate_field_path_elem(&mut self, layout: TyAndLayout<'tcx>, field: usize) -> PathElem {
        // First, check if we are projecting to a variant.
        match layout.variants {
            Variants::Multiple { tag_field, .. } => {
                if tag_field.as_usize() == field {
                    return match layout.ty.kind() {
                        ty::Adt(def, ..) if def.is_enum() => PathElem::EnumTag,
                        ty::Coroutine(..) => PathElem::CoroutineTag,
                        _ => bug!("non-variant type {:?}", layout.ty),
                    };
                }
            }
            Variants::Single { .. } | Variants::Empty => {}
        }

        // Now we know we are projecting to a field, so figure out which one.
        match layout.ty.kind() {
            // coroutines, closures, and coroutine-closures all have upvars that may be named.
            ty::Closure(def_id, _) | ty::Coroutine(def_id, _) | ty::CoroutineClosure(def_id, _) => {
                let mut name = None;
                // FIXME this should be more descriptive i.e. CapturePlace instead of CapturedVar
                // https://github.com/rust-lang/project-rfc-2229/issues/46
                if let Some(local_def_id) = def_id.as_local() {
                    let captures = self.ecx.tcx.closure_captures(local_def_id);
                    if let Some(captured_place) = captures.get(field) {
                        // Sometimes the index is beyond the number of upvars (seen
                        // for a coroutine).
                        let var_hir_id = captured_place.get_root_variable();
                        let node = self.ecx.tcx.hir_node(var_hir_id);
                        if let hir::Node::Pat(pat) = node {
                            if let hir::PatKind::Binding(_, _, ident, _) = pat.kind {
                                name = Some(ident.name);
                            }
                        }
                    }
                }

                PathElem::CapturedVar(name.unwrap_or_else(|| {
                    // Fall back to showing the field index.
                    sym::integer(field)
                }))
            }

            // tuples
            ty::Tuple(_) => PathElem::TupleElem(field),

            // enums
            ty::Adt(def, ..) if def.is_enum() => {
                // we might be projecting *to* a variant, or to a field *in* a variant.
                match layout.variants {
                    Variants::Single { index } => {
                        // Inside a variant
                        PathElem::Field(def.variant(index).fields[FieldIdx::from_usize(field)].name)
                    }
                    Variants::Empty => panic!("there is no field in Variants::Empty types"),
                    Variants::Multiple { .. } => bug!("we handled variants above"),
                }
            }

            // other ADTs
            ty::Adt(def, _) => {
                PathElem::Field(def.non_enum_variant().fields[FieldIdx::from_usize(field)].name)
            }

            // arrays/slices
            ty::Array(..) | ty::Slice(..) => PathElem::ArrayElem(field),

            // dyn* vtables
            ty::Dynamic(_, _, ty::DynKind::DynStar) if field == 1 => PathElem::Vtable,

            // dyn traits
            ty::Dynamic(..) => {
                assert_eq!(field, 0);
                PathElem::DynDowncast
            }

            // nothing else has an aggregate layout
            _ => bug!("aggregate_field_path_elem: got non-aggregate type {:?}", layout.ty),
        }
    }

    fn with_elem<R>(
        &mut self,
        elem: PathElem,
        f: impl FnOnce(&mut Self) -> InterpResult<'tcx, R>,
    ) -> InterpResult<'tcx, R> {
        // Remember the old state
        let path_len = self.path.len();
        // Record new element
        self.path.push(elem);
        // Perform operation
        let r = f(self)?;
        // Undo changes
        self.path.truncate(path_len);
        // Done
        interp_ok(r)
    }

    fn read_immediate(
        &self,
        val: &PlaceTy<'tcx, M::Provenance>,
        expected: ExpectedKind,
    ) -> InterpResult<'tcx, ImmTy<'tcx, M::Provenance>> {
        interp_ok(try_validation!(
            self.ecx.read_immediate(val),
            self.path,
            Ub(InvalidUninitBytes(None)) =>
                Uninit { expected },
            // The `Unsup` cases can only occur during CTFE
            Unsup(ReadPointerAsInt(_)) =>
                PointerAsInt { expected },
            Unsup(ReadPartialPointer(_)) =>
                PartialPointer,
        ))
    }

    fn read_scalar(
        &self,
        val: &PlaceTy<'tcx, M::Provenance>,
        expected: ExpectedKind,
    ) -> InterpResult<'tcx, Scalar<M::Provenance>> {
        interp_ok(self.read_immediate(val, expected)?.to_scalar())
    }

    fn deref_pointer(
        &mut self,
        val: &PlaceTy<'tcx, M::Provenance>,
        expected: ExpectedKind,
    ) -> InterpResult<'tcx, MPlaceTy<'tcx, M::Provenance>> {
        // Not using `ecx.deref_pointer` since we want to use our `read_immediate` wrapper.
        let imm = self.read_immediate(val, expected)?;
        // Reset provenance: ensure slice tail metadata does not preserve provenance,
        // and ensure all pointers do not preserve partial provenance.
        if self.reset_provenance_and_padding {
            if matches!(imm.layout.backend_repr, BackendRepr::Scalar(..)) {
                // A thin pointer. If it has provenance, we don't have to do anything.
                // If it does not, ensure we clear the provenance in memory.
                if matches!(imm.to_scalar(), Scalar::Int(..)) {
                    self.ecx.clear_provenance(val)?;
                }
            } else {
                // A wide pointer. This means we have to worry both about the pointer itself and the
                // metadata. We do the lazy thing and just write back the value we got. Just
                // clearing provenance in a targeted manner would be more efficient, but unless this
                // is a perf hotspot it's just not worth the effort.
                self.ecx.write_immediate_no_validate(*imm, val)?;
            }
            // The entire thing is data, not padding.
            self.add_data_range_place(val);
        }
        // Now turn it into a place.
        self.ecx.ref_to_mplace(&imm)
    }

    fn check_wide_ptr_meta(
        &mut self,
        meta: MemPlaceMeta<M::Provenance>,
        pointee: TyAndLayout<'tcx>,
    ) -> InterpResult<'tcx> {
        let tail = self.ecx.tcx.struct_tail_for_codegen(pointee.ty, self.ecx.typing_env);
        match tail.kind() {
            ty::Dynamic(data, _, ty::Dyn) => {
                let vtable = meta.unwrap_meta().to_pointer(self.ecx)?;
                // Make sure it is a genuine vtable pointer for the right trait.
                try_validation!(
                    self.ecx.get_ptr_vtable_ty(vtable, Some(data)),
                    self.path,
                    Ub(DanglingIntPointer{ .. } | InvalidVTablePointer(..)) =>
                        InvalidVTablePtr { value: format!("{vtable}") },
                    Ub(InvalidVTableTrait { vtable_dyn_type, expected_dyn_type }) => {
                        InvalidMetaWrongTrait { vtable_dyn_type, expected_dyn_type }
                    },
                );
            }
            ty::Slice(..) | ty::Str => {
                let _len = meta.unwrap_meta().to_target_usize(self.ecx)?;
                // We do not check that `len * elem_size <= isize::MAX`:
                // that is only required for references, and there it falls out of the
                // "dereferenceable" check performed by Stacked Borrows.
            }
            ty::Foreign(..) => {
                // Unsized, but not wide.
            }
            _ => bug!("Unexpected unsized type tail: {:?}", tail),
        }

        interp_ok(())
    }

    /// Check a reference or `Box`.
    fn check_safe_pointer(
        &mut self,
        value: &PlaceTy<'tcx, M::Provenance>,
        ptr_kind: PointerKind,
    ) -> InterpResult<'tcx> {
        let place = self.deref_pointer(value, ptr_kind.into())?;
        // Handle wide pointers.
        // Check metadata early, for better diagnostics
        if place.layout.is_unsized() {
            self.check_wide_ptr_meta(place.meta(), place.layout)?;
        }
        // Make sure this is dereferenceable and all.
        let size_and_align = try_validation!(
            self.ecx.size_and_align_of_val(&place),
            self.path,
            Ub(InvalidMeta(msg)) => match msg {
                InvalidMetaKind::SliceTooBig => InvalidMetaSliceTooLarge { ptr_kind },
                InvalidMetaKind::TooBig => InvalidMetaTooLarge { ptr_kind },
            }
        );
        let (size, align) = size_and_align
            // for the purpose of validity, consider foreign types to have
            // alignment and size determined by the layout (size will be 0,
            // alignment should take attributes into account).
            .unwrap_or_else(|| (place.layout.size, place.layout.align.abi));
        // Direct call to `check_ptr_access_align` checks alignment even on CTFE machines.
        try_validation!(
            self.ecx.check_ptr_access(
                place.ptr(),
                size,
                CheckInAllocMsg::Dereferenceable, // will anyway be replaced by validity message
            ),
            self.path,
            Ub(DanglingIntPointer { addr: 0, .. }) => NullPtr { ptr_kind },
            Ub(DanglingIntPointer { addr: i, .. }) => DanglingPtrNoProvenance {
                ptr_kind,
                // FIXME this says "null pointer" when null but we need translate
                pointer: format!("{}", Pointer::<Option<AllocId>>::from_addr_invalid(i))
            },
            Ub(PointerOutOfBounds { .. }) => DanglingPtrOutOfBounds {
                ptr_kind
            },
            Ub(PointerUseAfterFree(..)) => DanglingPtrUseAfterFree {
                ptr_kind,
            },
        );
        try_validation!(
            self.ecx.check_ptr_align(
                place.ptr(),
                align,
            ),
            self.path,
            Ub(AlignmentCheckFailed(Misalignment { required, has }, _msg)) => UnalignedPtr {
                ptr_kind,
                required_bytes: required.bytes(),
                found_bytes: has.bytes()
            },
        );
        // Make sure this is non-null. We checked dereferenceability above, but if `size` is zero
        // that does not imply non-null.
        if self.ecx.scalar_may_be_null(Scalar::from_maybe_pointer(place.ptr(), self.ecx))? {
            throw_validation_failure!(self.path, NullPtr { ptr_kind })
        }
        // Do not allow references to uninhabited types.
        if place.layout.is_uninhabited() {
            let ty = place.layout.ty;
            throw_validation_failure!(self.path, PtrToUninhabited { ptr_kind, ty })
        }
        // Recursive checking
        if let Some(ref_tracking) = self.ref_tracking.as_deref_mut() {
            // Proceed recursively even for ZST, no reason to skip them!
            // `!` is a ZST and we want to validate it.
            if let Some(ctfe_mode) = self.ctfe_mode {
                let mut skip_recursive_check = false;
                // CTFE imposes restrictions on what references can point to.
                if let Ok((alloc_id, _offset, _prov)) =
                    self.ecx.ptr_try_get_alloc_id(place.ptr(), 0)
                {
                    // Everything should be already interned.
                    let Some(global_alloc) = self.ecx.tcx.try_get_global_alloc(alloc_id) else {
                        assert!(self.ecx.memory.alloc_map.get(alloc_id).is_none());
                        // We can't have *any* references to non-existing allocations in const-eval
                        // as the rest of rustc isn't happy with them... so we throw an error, even
                        // though for zero-sized references this isn't really UB.
                        // A potential future alternative would be to resurrect this as a zero-sized allocation
                        // (which codegen will then compile to an aligned dummy pointer anyway).
                        throw_validation_failure!(self.path, DanglingPtrUseAfterFree { ptr_kind });
                    };
                    let (size, _align) =
                        global_alloc.size_and_align(*self.ecx.tcx, self.ecx.typing_env);
                    let alloc_actual_mutbl =
                        global_alloc.mutability(*self.ecx.tcx, self.ecx.typing_env);

                    if let GlobalAlloc::Static(did) = global_alloc {
                        let DefKind::Static { nested, .. } = self.ecx.tcx.def_kind(did) else {
                            bug!()
                        };
                        // Special handling for pointers to statics (irrespective of their type).
                        assert!(!self.ecx.tcx.is_thread_local_static(did));
                        assert!(self.ecx.tcx.is_static(did));
                        // Mode-specific checks
                        match ctfe_mode {
                            CtfeValidationMode::Static { .. }
                            | CtfeValidationMode::Promoted { .. } => {
                                // We skip recursively checking other statics. These statics must be sound by
                                // themselves, and the only way to get broken statics here is by using
                                // unsafe code.
                                // The reasons we don't check other statics is twofold. For one, in all
                                // sound cases, the static was already validated on its own, and second, we
                                // trigger cycle errors if we try to compute the value of the other static
                                // and that static refers back to us (potentially through a promoted).
                                // This could miss some UB, but that's fine.
                                // We still walk nested allocations, as they are fundamentally part of this validation run.
                                // This means we will also recurse into nested statics of *other*
                                // statics, even though we do not recurse into other statics directly.
                                // That's somewhat inconsistent but harmless.
                                skip_recursive_check = !nested;
                            }
                            CtfeValidationMode::Const { .. } => {
                                // If this is mutable memory or an `extern static`, there's no point in checking it -- we'd
                                // just get errors trying to read the value.
                                if alloc_actual_mutbl.is_mut() || self.ecx.tcx.is_foreign_item(did)
                                {
                                    skip_recursive_check = true;
                                }
                            }
                        }
                    }

                    // If this allocation has size zero, there is no actual mutability here.
                    if size != Size::ZERO {
                        // Determine whether this pointer expects to be pointing to something mutable.
                        let ptr_expected_mutbl = match ptr_kind {
                            PointerKind::Box => Mutability::Mut,
                            PointerKind::Ref(mutbl) => {
                                // We do not take into account interior mutability here since we cannot know if
                                // there really is an `UnsafeCell` inside `Option<UnsafeCell>` -- so we check
                                // that in the recursive descent behind this reference (controlled by
                                // `allow_immutable_unsafe_cell`).
                                mutbl
                            }
                        };
                        // Mutable pointer to immutable memory is no good.
                        if ptr_expected_mutbl == Mutability::Mut
                            && alloc_actual_mutbl == Mutability::Not
                        {
                            // This can actually occur with transmutes.
                            throw_validation_failure!(self.path, MutableRefToImmutable);
                        }
                        // In a const, any kind of mutable reference is not good.
                        if matches!(self.ctfe_mode, Some(CtfeValidationMode::Const { .. })) {
                            if ptr_expected_mutbl == Mutability::Mut {
                                throw_validation_failure!(self.path, MutableRefInConst);
                            }
                        }
                    }
                }
                // Potentially skip recursive check.
                if skip_recursive_check {
                    return interp_ok(());
                }
            } else {
                // This is not CTFE, so it's Miri with recursive checking.
                // FIXME: we do *not* check behind boxes, since creating a new box first creates it uninitialized
                // and then puts the value in there, so briefly we have a box with uninit contents.
                // FIXME: should we also skip `UnsafeCell` behind shared references? Currently that is not
                // needed since validation reads bypass Stacked Borrows and data race checks.
                if matches!(ptr_kind, PointerKind::Box) {
                    return interp_ok(());
                }
            }
            let path = &self.path;
            ref_tracking.track(place, || {
                // We need to clone the path anyway, make sure it gets created
                // with enough space for the additional `Deref`.
                let mut new_path = Vec::with_capacity(path.len() + 1);
                new_path.extend(path);
                new_path.push(PathElem::Deref);
                new_path
            });
        }
        interp_ok(())
    }

    /// Check if this is a value of primitive type, and if yes check the validity of the value
    /// at that type. Return `true` if the type is indeed primitive.
    ///
    /// Note that not all of these have `FieldsShape::Primitive`, e.g. wide references.
    fn try_visit_primitive(
        &mut self,
        value: &PlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, bool> {
        // Go over all the primitive types
        let ty = value.layout.ty;
        match ty.kind() {
            ty::Bool => {
                let scalar = self.read_scalar(value, ExpectedKind::Bool)?;
                try_validation!(
                    scalar.to_bool(),
                    self.path,
                    Ub(InvalidBool(..)) => ValidationErrorKind::InvalidBool {
                        value: format!("{scalar:x}"),
                    }
                );
                if self.reset_provenance_and_padding {
                    self.ecx.clear_provenance(value)?;
                    self.add_data_range_place(value);
                }
                interp_ok(true)
            }
            ty::Char => {
                let scalar = self.read_scalar(value, ExpectedKind::Char)?;
                try_validation!(
                    scalar.to_char(),
                    self.path,
                    Ub(InvalidChar(..)) => ValidationErrorKind::InvalidChar {
                        value: format!("{scalar:x}"),
                    }
                );
                if self.reset_provenance_and_padding {
                    self.ecx.clear_provenance(value)?;
                    self.add_data_range_place(value);
                }
                interp_ok(true)
            }
            ty::Float(_) | ty::Int(_) | ty::Uint(_) => {
                // NOTE: Keep this in sync with the array optimization for int/float
                // types below!
                self.read_scalar(
                    value,
                    if matches!(ty.kind(), ty::Float(..)) {
                        ExpectedKind::Float
                    } else {
                        ExpectedKind::Int
                    },
                )?;
                if self.reset_provenance_and_padding {
                    self.ecx.clear_provenance(value)?;
                    self.add_data_range_place(value);
                }
                interp_ok(true)
            }
            ty::RawPtr(..) => {
                let place = self.deref_pointer(value, ExpectedKind::RawPtr)?;
                if place.layout.is_unsized() {
                    self.check_wide_ptr_meta(place.meta(), place.layout)?;
                }
                interp_ok(true)
            }
            ty::Ref(_, _ty, mutbl) => {
                self.check_safe_pointer(value, PointerKind::Ref(*mutbl))?;
                interp_ok(true)
            }
            ty::FnPtr(..) => {
                let scalar = self.read_scalar(value, ExpectedKind::FnPtr)?;

                // If we check references recursively, also check that this points to a function.
                if let Some(_) = self.ref_tracking {
                    let ptr = scalar.to_pointer(self.ecx)?;
                    let _fn = try_validation!(
                        self.ecx.get_ptr_fn(ptr),
                        self.path,
                        Ub(DanglingIntPointer{ .. } | InvalidFunctionPointer(..)) =>
                            InvalidFnPtr { value: format!("{ptr}") },
                    );
                    // FIXME: Check if the signature matches
                } else {
                    // Otherwise (for standalone Miri), we have to still check it to be non-null.
                    if self.ecx.scalar_may_be_null(scalar)? {
                        throw_validation_failure!(self.path, NullFnPtr);
                    }
                }
                if self.reset_provenance_and_padding {
                    // Make sure we do not preserve partial provenance. This matches the thin
                    // pointer handling in `deref_pointer`.
                    if matches!(scalar, Scalar::Int(..)) {
                        self.ecx.clear_provenance(value)?;
                    }
                    self.add_data_range_place(value);
                }
                interp_ok(true)
            }
            ty::Never => throw_validation_failure!(self.path, NeverVal),
            ty::Foreign(..) | ty::FnDef(..) => {
                // Nothing to check.
                interp_ok(true)
            }
            ty::UnsafeBinder(_) => todo!("FIXME(unsafe_binder)"),
            // The above should be all the primitive types. The rest is compound, we
            // check them by visiting their fields/variants.
            ty::Adt(..)
            | ty::Tuple(..)
            | ty::Array(..)
            | ty::Slice(..)
            | ty::Str
            | ty::Dynamic(..)
            | ty::Closure(..)
            | ty::Pat(..)
            | ty::CoroutineClosure(..)
            | ty::Coroutine(..) => interp_ok(false),
            // Some types only occur during typechecking, they have no layout.
            // We should not see them here and we could not check them anyway.
            ty::Error(_)
            | ty::Infer(..)
            | ty::Placeholder(..)
            | ty::Bound(..)
            | ty::Param(..)
            | ty::Alias(..)
            | ty::CoroutineWitness(..) => bug!("Encountered invalid type {:?}", ty),
        }
    }

    fn visit_scalar(
        &mut self,
        scalar: Scalar<M::Provenance>,
        scalar_layout: ScalarAbi,
    ) -> InterpResult<'tcx> {
        let size = scalar_layout.size(self.ecx);
        let valid_range = scalar_layout.valid_range(self.ecx);
        let WrappingRange { start, end } = valid_range;
        let max_value = size.unsigned_int_max();
        assert!(end <= max_value);
        let bits = match scalar.try_to_scalar_int() {
            Ok(int) => int.to_bits(size),
            Err(_) => {
                // So this is a pointer then, and casting to an int failed.
                // Can only happen during CTFE.
                // We support 2 kinds of ranges here: full range, and excluding zero.
                if start == 1 && end == max_value {
                    // Only null is the niche. So make sure the ptr is NOT null.
                    if self.ecx.scalar_may_be_null(scalar)? {
                        throw_validation_failure!(
                            self.path,
                            NullablePtrOutOfRange { range: valid_range, max_value }
                        )
                    } else {
                        return interp_ok(());
                    }
                } else if scalar_layout.is_always_valid(self.ecx) {
                    // Easy. (This is reachable if `enforce_number_validity` is set.)
                    return interp_ok(());
                } else {
                    // Conservatively, we reject, because the pointer *could* have a bad
                    // value.
                    throw_validation_failure!(
                        self.path,
                        PtrOutOfRange { range: valid_range, max_value }
                    )
                }
            }
        };
        // Now compare.
        if valid_range.contains(bits) {
            interp_ok(())
        } else {
            throw_validation_failure!(
                self.path,
                OutOfRange { value: format!("{bits}"), range: valid_range, max_value }
            )
        }
    }

    fn in_mutable_memory(&self, val: &PlaceTy<'tcx, M::Provenance>) -> bool {
        debug_assert!(self.ctfe_mode.is_some());
        if let Some(mplace) = val.as_mplace_or_local().left() {
            if let Some(alloc_id) = mplace.ptr().provenance.and_then(|p| p.get_alloc_id()) {
                let tcx = *self.ecx.tcx;
                // Everything must be already interned.
                let mutbl = tcx.global_alloc(alloc_id).mutability(tcx, self.ecx.typing_env);
                if let Some((_, alloc)) = self.ecx.memory.alloc_map.get(alloc_id) {
                    assert_eq!(alloc.mutability, mutbl);
                }
                mutbl.is_mut()
            } else {
                // No memory at all.
                false
            }
        } else {
            // A local variable -- definitely mutable.
            true
        }
    }

    /// Add the given pointer-length pair to the "data" range of this visit.
    fn add_data_range(&mut self, ptr: Pointer<Option<M::Provenance>>, size: Size) {
        if let Some(data_bytes) = self.data_bytes.as_mut() {
            // We only have to store the offset, the rest is the same for all pointers here.
            let (_prov, offset) = ptr.into_parts();
            // Add this.
            data_bytes.add_range(offset, size);
        };
    }

    /// Add the entire given place to the "data" range of this visit.
    fn add_data_range_place(&mut self, place: &PlaceTy<'tcx, M::Provenance>) {
        // Only sized places can be added this way.
        debug_assert!(place.layout.is_sized());
        if let Some(data_bytes) = self.data_bytes.as_mut() {
            let offset = Self::data_range_offset(self.ecx, place);
            data_bytes.add_range(offset, place.layout.size);
        }
    }

    /// Convert a place into the offset it starts at, for the purpose of data_range tracking.
    /// Must only be called if `data_bytes` is `Some(_)`.
    fn data_range_offset(ecx: &InterpCx<'tcx, M>, place: &PlaceTy<'tcx, M::Provenance>) -> Size {
        // The presence of `data_bytes` implies that our place is in memory.
        let ptr = ecx
            .place_to_op(place)
            .expect("place must be in memory")
            .as_mplace_or_imm()
            .expect_left("place must be in memory")
            .ptr();
        let (_prov, offset) = ptr.into_parts();
        offset
    }

    fn reset_padding(&mut self, place: &PlaceTy<'tcx, M::Provenance>) -> InterpResult<'tcx> {
        let Some(data_bytes) = self.data_bytes.as_mut() else { return interp_ok(()) };
        // Our value must be in memory, otherwise we would not have set up `data_bytes`.
        let mplace = self.ecx.force_allocation(place)?;
        // Determine starting offset and size.
        let (_prov, start_offset) = mplace.ptr().into_parts();
        let (size, _align) = self
            .ecx
            .size_and_align_of_val(&mplace)?
            .unwrap_or((mplace.layout.size, mplace.layout.align.abi));
        // If there is no padding at all, we can skip the rest: check for
        // a single data range covering the entire value.
        if data_bytes.0 == &[(start_offset, size)] {
            return interp_ok(());
        }
        // Get a handle for the allocation. Do this only once, to avoid looking up the same
        // allocation over and over again. (Though to be fair, iterating the value already does
        // exactly that.)
        let Some(mut alloc) = self.ecx.get_ptr_alloc_mut(mplace.ptr(), size)? else {
            // A ZST, no padding to clear.
            return interp_ok(());
        };
        // Add a "finalizer" data range at the end, so that the iteration below finds all gaps
        // between ranges.
        data_bytes.0.push((start_offset + size, Size::ZERO));
        // Iterate, and reset gaps.
        let mut padding_cleared_until = start_offset;
        for &(offset, size) in data_bytes.0.iter() {
            assert!(
                offset >= padding_cleared_until,
                "reset_padding on {}: previous field ended at offset {}, next field starts at {} (and has a size of {} bytes)",
                mplace.layout.ty,
                (padding_cleared_until - start_offset).bytes(),
                (offset - start_offset).bytes(),
                size.bytes(),
            );
            if offset > padding_cleared_until {
                // We found padding. Adjust the range to be relative to `alloc`, and make it uninit.
                let padding_start = padding_cleared_until - start_offset;
                let padding_size = offset - padding_cleared_until;
                let range = alloc_range(padding_start, padding_size);
                trace!("reset_padding on {}: resetting padding range {range:?}", mplace.layout.ty);
                alloc.write_uninit(range)?;
            }
            padding_cleared_until = offset + size;
        }
        assert!(padding_cleared_until == start_offset + size);
        interp_ok(())
    }

    /// Computes the data range of this union type:
    /// which bytes are inside a field (i.e., not padding.)
    fn union_data_range<'e>(
        ecx: &'e mut InterpCx<'tcx, M>,
        layout: TyAndLayout<'tcx>,
    ) -> Cow<'e, RangeSet> {
        assert!(layout.ty.is_union());
        assert!(layout.is_sized(), "there are no unsized unions");
        let layout_cx = LayoutCx::new(*ecx.tcx, ecx.typing_env);
        return M::cached_union_data_range(ecx, layout.ty, || {
            let mut out = RangeSet(Vec::new());
            union_data_range_uncached(&layout_cx, layout, Size::ZERO, &mut out);
            out
        });

        /// Helper for recursive traversal: add data ranges of the given type to `out`.
        fn union_data_range_uncached<'tcx>(
            cx: &LayoutCx<'tcx>,
            layout: TyAndLayout<'tcx>,
            base_offset: Size,
            out: &mut RangeSet,
        ) {
            // If this is a ZST, we don't contain any data. In particular, this helps us to quickly
            // skip over huge arrays of ZST.
            if layout.is_zst() {
                return;
            }
            // Just recursively add all the fields of everything to the output.
            match &layout.fields {
                FieldsShape::Primitive => {
                    out.add_range(base_offset, layout.size);
                }
                &FieldsShape::Union(fields) => {
                    // Currently, all fields start at offset 0 (relative to `base_offset`).
                    for field in 0..fields.get() {
                        let field = layout.field(cx, field);
                        union_data_range_uncached(cx, field, base_offset, out);
                    }
                }
                &FieldsShape::Array { stride, count } => {
                    let elem = layout.field(cx, 0);

                    // Fast-path for large arrays of simple types that do not contain any padding.
                    if elem.backend_repr.is_scalar() {
                        out.add_range(base_offset, elem.size * count);
                    } else {
                        for idx in 0..count {
                            // This repeats the same computation for every array element... but the alternative
                            // is to allocate temporary storage for a dedicated `out` set for the array element,
                            // and replicating that N times. Is that better?
                            union_data_range_uncached(cx, elem, base_offset + idx * stride, out);
                        }
                    }
                }
                FieldsShape::Arbitrary { offsets, .. } => {
                    for (field, &offset) in offsets.iter_enumerated() {
                        let field = layout.field(cx, field.as_usize());
                        union_data_range_uncached(cx, field, base_offset + offset, out);
                    }
                }
            }
            // Don't forget potential other variants.
            match &layout.variants {
                Variants::Single { .. } | Variants::Empty => {
                    // Fully handled above.
                }
                Variants::Multiple { variants, .. } => {
                    for variant in variants.indices() {
                        let variant = layout.for_variant(cx, variant);
                        union_data_range_uncached(cx, variant, base_offset, out);
                    }
                }
            }
        }
    }
}

impl<'rt, 'tcx, M: Machine<'tcx>> ValueVisitor<'tcx, M> for ValidityVisitor<'rt, 'tcx, M> {
    type V = PlaceTy<'tcx, M::Provenance>;

    #[inline(always)]
    fn ecx(&self) -> &InterpCx<'tcx, M> {
        self.ecx
    }

    fn read_discriminant(
        &mut self,
        val: &PlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, VariantIdx> {
        self.with_elem(PathElem::EnumTag, move |this| {
            interp_ok(try_validation!(
                this.ecx.read_discriminant(val),
                this.path,
                Ub(InvalidTag(val)) => InvalidEnumTag {
                    value: format!("{val:x}"),
                },
                Ub(UninhabitedEnumVariantRead(_)) => UninhabitedEnumVariant,
                // Uninit / bad provenance are not possible since the field was already previously
                // checked at its integer type.
            ))
        })
    }

    #[inline]
    fn visit_field(
        &mut self,
        old_val: &PlaceTy<'tcx, M::Provenance>,
        field: usize,
        new_val: &PlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx> {
        let elem = self.aggregate_field_path_elem(old_val.layout, field);
        self.with_elem(elem, move |this| this.visit_value(new_val))
    }

    #[inline]
    fn visit_variant(
        &mut self,
        old_val: &PlaceTy<'tcx, M::Provenance>,
        variant_id: VariantIdx,
        new_val: &PlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx> {
        let name = match old_val.layout.ty.kind() {
            ty::Adt(adt, _) => PathElem::Variant(adt.variant(variant_id).name),
            // Coroutines also have variants
            ty::Coroutine(..) => PathElem::CoroutineState(variant_id),
            _ => bug!("Unexpected type with variant: {:?}", old_val.layout.ty),
        };
        self.with_elem(name, move |this| this.visit_value(new_val))
    }

    #[inline(always)]
    fn visit_union(
        &mut self,
        val: &PlaceTy<'tcx, M::Provenance>,
        _fields: NonZero<usize>,
    ) -> InterpResult<'tcx> {
        // Special check for CTFE validation, preventing `UnsafeCell` inside unions in immutable memory.
        if self.ctfe_mode.is_some_and(|c| !c.allow_immutable_unsafe_cell()) {
            // Unsized unions are currently not a thing, but let's keep this code consistent with
            // the check in `visit_value`.
            let zst = self.ecx.size_and_align_of_val(val)?.is_some_and(|(s, _a)| s.bytes() == 0);
            if !zst && !val.layout.ty.is_freeze(*self.ecx.tcx, self.ecx.typing_env) {
                if !self.in_mutable_memory(val) {
                    throw_validation_failure!(self.path, UnsafeCellInImmutable);
                }
            }
        }
        if self.reset_provenance_and_padding
            && let Some(data_bytes) = self.data_bytes.as_mut()
        {
            let base_offset = Self::data_range_offset(self.ecx, val);
            // Determine and add data range for this union.
            let union_data_range = Self::union_data_range(self.ecx, val.layout);
            for &(offset, size) in union_data_range.0.iter() {
                data_bytes.add_range(base_offset + offset, size);
            }
        }
        interp_ok(())
    }

    #[inline]
    fn visit_box(
        &mut self,
        _box_ty: Ty<'tcx>,
        val: &PlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx> {
        self.check_safe_pointer(val, PointerKind::Box)?;
        interp_ok(())
    }

    #[inline]
    fn visit_value(&mut self, val: &PlaceTy<'tcx, M::Provenance>) -> InterpResult<'tcx> {
        trace!("visit_value: {:?}, {:?}", *val, val.layout);

        // Check primitive types -- the leaves of our recursive descent.
        // This is called even for enum discriminants (which are "fields" of their enum),
        // so for integer-typed discriminants the provenance reset will happen here.
        // We assume that the Scalar validity range does not restrict these values
        // any further than `try_visit_primitive` does!
        if self.try_visit_primitive(val)? {
            return interp_ok(());
        }

        // Special check preventing `UnsafeCell` in the inner part of constants
        if self.ctfe_mode.is_some_and(|c| !c.allow_immutable_unsafe_cell()) {
            // Exclude ZST values. We need to compute the dynamic size/align to properly
            // handle slices and trait objects.
            let zst = self.ecx.size_and_align_of_val(val)?.is_some_and(|(s, _a)| s.bytes() == 0);
            if !zst
                && let Some(def) = val.layout.ty.ty_adt_def()
                && def.is_unsafe_cell()
            {
                if !self.in_mutable_memory(val) {
                    throw_validation_failure!(self.path, UnsafeCellInImmutable);
                }
            }
        }

        // Recursively walk the value at its type. Apply optimizations for some large types.
        match val.layout.ty.kind() {
            ty::Str => {
                let mplace = val.assert_mem_place(); // strings are unsized and hence never immediate
                let len = mplace.len(self.ecx)?;
                try_validation!(
                    self.ecx.read_bytes_ptr_strip_provenance(mplace.ptr(), Size::from_bytes(len)),
                    self.path,
                    Ub(InvalidUninitBytes(..)) => Uninit { expected: ExpectedKind::Str },
                    Unsup(ReadPointerAsInt(_)) => PointerAsInt { expected: ExpectedKind::Str }
                );
            }
            ty::Array(tys, ..) | ty::Slice(tys)
                // This optimization applies for types that can hold arbitrary non-provenance bytes (such as
                // integer and floating point types).
                // FIXME(wesleywiser) This logic could be extended further to arbitrary structs or
                // tuples made up of integer/floating point types or inhabited ZSTs with no padding.
                if matches!(tys.kind(), ty::Int(..) | ty::Uint(..) | ty::Float(..))
                =>
            {
                let expected = if tys.is_integral() { ExpectedKind::Int } else { ExpectedKind::Float };
                // Optimized handling for arrays of integer/float type.

                // This is the length of the array/slice.
                let len = val.len(self.ecx)?;
                // This is the element type size.
                let layout = self.ecx.layout_of(*tys)?;
                // This is the size in bytes of the whole array. (This checks for overflow.)
                let size = layout.size * len;
                // If the size is 0, there is nothing to check.
                // (`size` can only be 0 if `len` is 0, and empty arrays are always valid.)
                if size == Size::ZERO {
                    return interp_ok(());
                }
                // Now that we definitely have a non-ZST array, we know it lives in memory -- except it may
                // be an uninitialized local variable, those are also "immediate".
                let mplace = match val.to_op(self.ecx)?.as_mplace_or_imm() {
                    Left(mplace) => mplace,
                    Right(imm) => match *imm {
                        Immediate::Uninit =>
                            throw_validation_failure!(self.path, Uninit { expected }),
                        Immediate::Scalar(..) | Immediate::ScalarPair(..) =>
                            bug!("arrays/slices can never have Scalar/ScalarPair layout"),
                    }
                };

                // Optimization: we just check the entire range at once.
                // NOTE: Keep this in sync with the handling of integer and float
                // types above, in `visit_primitive`.
                // No need for an alignment check here, this is not an actual memory access.
                let alloc = self.ecx.get_ptr_alloc(mplace.ptr(), size)?.expect("we already excluded size 0");

                alloc.get_bytes_strip_provenance().map_err_kind(|kind| {
                    // Some error happened, try to provide a more detailed description.
                    // For some errors we might be able to provide extra information.
                    // (This custom logic does not fit the `try_validation!` macro.)
                    match kind {
                        Ub(InvalidUninitBytes(Some((_alloc_id, access)))) | Unsup(ReadPointerAsInt(Some((_alloc_id, access)))) => {
                            // Some byte was uninitialized, determine which
                            // element that byte belongs to so we can
                            // provide an index.
                            let i = usize::try_from(
                                access.bad.start.bytes() / layout.size.bytes(),
                            )
                            .unwrap();
                            self.path.push(PathElem::ArrayElem(i));

                            if matches!(kind, Ub(InvalidUninitBytes(_))) {
                                err_validation_failure!(self.path, Uninit { expected })
                            } else {
                                err_validation_failure!(self.path, PointerAsInt { expected })
                            }
                        }

                        // Propagate upwards (that will also check for unexpected errors).
                        err => err,
                    }
                })?;

                // Don't forget that these are all non-pointer types, and thus do not preserve
                // provenance.
                if self.reset_provenance_and_padding {
                    // We can't share this with above as above, we might be looking at read-only memory.
                    let mut alloc = self.ecx.get_ptr_alloc_mut(mplace.ptr(), size)?.expect("we already excluded size 0");
                    alloc.clear_provenance()?;
                    // Also, mark this as containing data, not padding.
                    self.add_data_range(mplace.ptr(), size);
                }
            }
            // Fast path for arrays and slices of ZSTs. We only need to check a single ZST element
            // of an array and not all of them, because there's only a single value of a specific
            // ZST type, so either validation fails for all elements or none.
            ty::Array(tys, ..) | ty::Slice(tys) if self.ecx.layout_of(*tys)?.is_zst() => {
                // Validate just the first element (if any).
                if val.len(self.ecx)? > 0 {
                    self.visit_field(val, 0, &self.ecx.project_index(val, 0)?)?;
                }
            }
            ty::Pat(base, pat) => {
                // First check that the base type is valid
                self.visit_value(&val.transmute(self.ecx.layout_of(*base)?, self.ecx)?)?;
                // When you extend this match, make sure to also add tests to
                // tests/ui/type/pattern_types/validity.rs((
                match **pat {
                    // Range patterns are precisely reflected into `valid_range` and thus
                    // handled fully by `visit_scalar` (called below).
                    ty::PatternKind::Range { .. } => {},

                    // FIXME(pattern_types): check that the value is covered by one of the variants.
                    // For now, we rely on layout computation setting the scalar's `valid_range` to
                    // match the pattern. However, this cannot always work; the layout may
                    // pessimistically cover actually illegal ranges and Miri would miss that UB.
                    // The consolation here is that codegen also will miss that UB, so at least
                    // we won't see optimizations actually breaking such programs.
                    ty::PatternKind::Or(_patterns) => {}
                }
            }
            _ => {
                // default handler
                try_validation!(
                    self.walk_value(val),
                    self.path,
                    // It's not great to catch errors here, since we can't give a very good path,
                    // but it's better than ICEing.
                    Ub(InvalidVTableTrait { vtable_dyn_type, expected_dyn_type }) => {
                        InvalidMetaWrongTrait { vtable_dyn_type, expected_dyn_type }
                    },
                );
            }
        }

        // *After* all of this, check further information stored in the layout. We need to check
        // this to handle types like `NonNull` where the `Scalar` info is more restrictive than what
        // the fields say (`rustc_layout_scalar_valid_range_start`). But in most cases, this will
        // just propagate what the fields say, and then we want the error to point at the field --
        // so, we first recurse, then we do this check.
        //
        // FIXME: We could avoid some redundant checks here. For newtypes wrapping
        // scalars, we do the same check on every "level" (e.g., first we check
        // MyNewtype and then the scalar in there).
        if val.layout.is_uninhabited() {
            let ty = val.layout.ty;
            throw_validation_failure!(self.path, UninhabitedVal { ty });
        }
        match val.layout.backend_repr {
            BackendRepr::Scalar(scalar_layout) => {
                if !scalar_layout.is_uninit_valid() {
                    // There is something to check here.
                    let scalar = self.read_scalar(val, ExpectedKind::InitScalar)?;
                    self.visit_scalar(scalar, scalar_layout)?;
                }
            }
            BackendRepr::ScalarPair(a_layout, b_layout) => {
                // We can only proceed if *both* scalars need to be initialized.
                // FIXME: find a way to also check ScalarPair when one side can be uninit but
                // the other must be init.
                if !a_layout.is_uninit_valid() && !b_layout.is_uninit_valid() {
                    let (a, b) =
                        self.read_immediate(val, ExpectedKind::InitScalar)?.to_scalar_pair();
                    self.visit_scalar(a, a_layout)?;
                    self.visit_scalar(b, b_layout)?;
                }
            }
            BackendRepr::SimdVector { .. } => {
                // No checks here, we assume layout computation gets this right.
                // (This is harder to check since Miri does not represent these as `Immediate`. We
                // also cannot use field projections since this might be a newtype around a vector.)
            }
            BackendRepr::Memory { .. } => {
                // Nothing to do.
            }
        }

        interp_ok(())
    }
}

impl<'tcx, M: Machine<'tcx>> InterpCx<'tcx, M> {
    fn validate_operand_internal(
        &mut self,
        val: &PlaceTy<'tcx, M::Provenance>,
        path: Vec<PathElem>,
        ref_tracking: Option<&mut RefTracking<MPlaceTy<'tcx, M::Provenance>, Vec<PathElem>>>,
        ctfe_mode: Option<CtfeValidationMode>,
        reset_provenance_and_padding: bool,
    ) -> InterpResult<'tcx> {
        trace!("validate_operand_internal: {:?}, {:?}", *val, val.layout.ty);

        // Run the visitor.
        self.run_for_validation_mut(|ecx| {
            let reset_padding = reset_provenance_and_padding && {
                // Check if `val` is actually stored in memory. If not, padding is not even
                // represented and we need not reset it.
                ecx.place_to_op(val)?.as_mplace_or_imm().is_left()
            };
            let mut v = ValidityVisitor {
                path,
                ref_tracking,
                ctfe_mode,
                ecx,
                reset_provenance_and_padding,
                data_bytes: reset_padding.then_some(RangeSet(Vec::new())),
            };
            v.visit_value(val)?;
            v.reset_padding(val)?;
            interp_ok(())
        })
        .map_err_info(|err| {
            if !matches!(
                err.kind(),
                err_ub!(ValidationError { .. })
                    | InterpErrorKind::InvalidProgram(_)
                    | InterpErrorKind::Unsupported(UnsupportedOpInfo::ExternTypeField)
            ) {
                bug!(
                    "Unexpected error during validation: {}",
                    format_interp_error(self.tcx.dcx(), err)
                );
            }
            err
        })
    }

    /// This function checks the data at `val` to be const-valid.
    /// `val` is assumed to cover valid memory if it is an indirect operand.
    /// It will error if the bits at the destination do not match the ones described by the layout.
    ///
    /// `ref_tracking` is used to record references that we encounter so that they
    /// can be checked recursively by an outside driving loop.
    ///
    /// `constant` controls whether this must satisfy the rules for constants:
    /// - no pointers to statics.
    /// - no `UnsafeCell` or non-ZST `&mut`.
    #[inline(always)]
    pub(crate) fn const_validate_operand(
        &mut self,
        val: &PlaceTy<'tcx, M::Provenance>,
        path: Vec<PathElem>,
        ref_tracking: &mut RefTracking<MPlaceTy<'tcx, M::Provenance>, Vec<PathElem>>,
        ctfe_mode: CtfeValidationMode,
    ) -> InterpResult<'tcx> {
        self.validate_operand_internal(
            val,
            path,
            Some(ref_tracking),
            Some(ctfe_mode),
            /*reset_provenance*/ false,
        )
    }

    /// This function checks the data at `val` to be runtime-valid.
    /// `val` is assumed to cover valid memory if it is an indirect operand.
    /// It will error if the bits at the destination do not match the ones described by the layout.
    #[inline(always)]
    pub fn validate_operand(
        &mut self,
        val: &PlaceTy<'tcx, M::Provenance>,
        recursive: bool,
        reset_provenance_and_padding: bool,
    ) -> InterpResult<'tcx> {
        let _span = enter_trace_span!(
            M,
            "validate_operand",
            "recursive={recursive}, reset_provenance_and_padding={reset_provenance_and_padding}, val={val:?}"
        );

        // Note that we *could* actually be in CTFE here with `-Zextra-const-ub-checks`, but it's
        // still correct to not use `ctfe_mode`: that mode is for validation of the final constant
        // value, it rules out things like `UnsafeCell` in awkward places.
        if !recursive {
            return self.validate_operand_internal(
                val,
                vec![],
                None,
                None,
                reset_provenance_and_padding,
            );
        }
        // Do a recursive check.
        let mut ref_tracking = RefTracking::empty();
        self.validate_operand_internal(
            val,
            vec![],
            Some(&mut ref_tracking),
            None,
            reset_provenance_and_padding,
        )?;
        while let Some((mplace, path)) = ref_tracking.todo.pop() {
            // Things behind reference do *not* have the provenance reset.
            self.validate_operand_internal(
                &mplace.into(),
                path,
                Some(&mut ref_tracking),
                None,
                /*reset_provenance_and_padding*/ false,
            )?;
        }
        interp_ok(())
    }
}
