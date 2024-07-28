//! Check the validity invariant of a given value, and tell the user
//! where in the value it got violated.
//! In const context, this goes even further and tries to approximate const safety.
//! That's useful because it means other passes (e.g. promotion) can rely on `const`s
//! to be const-safe.

use std::fmt::Write;
use std::hash::Hash;
use std::num::NonZero;

use either::{Left, Right};
use tracing::trace;

use hir::def::DefKind;
use rustc_ast::Mutability;
use rustc_data_structures::fx::FxHashSet;
use rustc_hir as hir;
use rustc_middle::bug;
use rustc_middle::mir::interpret::{
    ExpectedKind, InterpError, InvalidMetaKind, Misalignment, PointerKind, Provenance,
    UnsupportedOpInfo, ValidationErrorInfo,
    ValidationErrorKind::{self, *},
};
use rustc_middle::ty::layout::{LayoutOf, TyAndLayout};
use rustc_middle::ty::{self, Ty};
use rustc_span::symbol::{sym, Symbol};
use rustc_target::abi::{
    Abi, FieldIdx, Scalar as ScalarAbi, Size, VariantIdx, Variants, WrappingRange,
};

use super::{
    err_ub, format_interp_error, machine::AllocMap, throw_ub, AllocId, AllocKind, CheckInAllocMsg,
    GlobalAlloc, ImmTy, Immediate, InterpCx, InterpResult, MPlaceTy, Machine, MemPlaceMeta, OpTy,
    Pointer, Projectable, Scalar, ValueVisitor,
};

// for the validation errors
#[rustfmt::skip]
use super::InterpError::UndefinedBehavior as Ub;
use super::InterpError::Unsupported as Unsup;
use super::UndefinedBehaviorInfo::*;
use super::UnsupportedOpInfo::*;

macro_rules! throw_validation_failure {
    ($where:expr, $kind: expr) => {{
        let where_ = &$where;
        let path = if !where_.is_empty() {
            let mut path = String::new();
            write_path(&mut path, where_);
            Some(path)
        } else {
            None
        };

        throw_ub!(ValidationError(ValidationErrorInfo { path, kind: $kind }))
    }};
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
        match $e {
            Ok(x) => x,
            // We catch the error and turn it into a validation failure. We are okay with
            // allocation here as this can only slow down builds that fail anyway.
            Err(e) => match e.kind() {
                $(
                    $($p)|+ =>
                       throw_validation_failure!(
                            $where,
                            $kind
                        )
                ),+,
                #[allow(unreachable_patterns)]
                _ => Err::<!, _>(e)?,
            }
        }
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
    pub seen: FxHashSet<T>,
    pub todo: Vec<(T, PATH)>,
}

impl<T: Clone + Eq + Hash + std::fmt::Debug, PATH: Default> RefTracking<T, PATH> {
    pub fn empty() -> Self {
        RefTracking { seen: FxHashSet::default(), todo: vec![] }
    }
    pub fn new(op: T) -> Self {
        let mut ref_tracking_for_consts =
            RefTracking { seen: FxHashSet::default(), todo: vec![(op.clone(), PATH::default())] };
        ref_tracking_for_consts.seen.insert(op);
        ref_tracking_for_consts
    }

    pub fn track(&mut self, op: T, path: impl FnOnce() -> PATH) {
        if self.seen.insert(op.clone()) {
            trace!("Recursing below ptr {:#?}", op);
            let path = path();
            // Remember to come back to this later.
            self.todo.push((op, path));
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
        }
        .unwrap()
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
    ecx: &'rt InterpCx<'tcx, M>,
}

impl<'rt, 'tcx, M: Machine<'tcx>> ValidityVisitor<'rt, 'tcx, M> {
    fn aggregate_field_path_elem(&mut self, layout: TyAndLayout<'tcx>, field: usize) -> PathElem {
        // First, check if we are projecting to a variant.
        match layout.variants {
            Variants::Multiple { tag_field, .. } => {
                if tag_field == field {
                    return match layout.ty.kind() {
                        ty::Adt(def, ..) if def.is_enum() => PathElem::EnumTag,
                        ty::Coroutine(..) => PathElem::CoroutineTag,
                        _ => bug!("non-variant type {:?}", layout.ty),
                    };
                }
            }
            Variants::Single { .. } => {}
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
                    Variants::Multiple { .. } => bug!("we handled variants above"),
                }
            }

            // other ADTs
            ty::Adt(def, _) => {
                PathElem::Field(def.non_enum_variant().fields[FieldIdx::from_usize(field)].name)
            }

            // arrays/slices
            ty::Array(..) | ty::Slice(..) => PathElem::ArrayElem(field),

            // dyn traits
            ty::Dynamic(..) => PathElem::DynDowncast,

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
        Ok(r)
    }

    fn read_immediate(
        &self,
        op: &OpTy<'tcx, M::Provenance>,
        expected: ExpectedKind,
    ) -> InterpResult<'tcx, ImmTy<'tcx, M::Provenance>> {
        Ok(try_validation!(
            self.ecx.read_immediate(op),
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
        op: &OpTy<'tcx, M::Provenance>,
        expected: ExpectedKind,
    ) -> InterpResult<'tcx, Scalar<M::Provenance>> {
        Ok(self.read_immediate(op, expected)?.to_scalar())
    }

    fn check_wide_ptr_meta(
        &mut self,
        meta: MemPlaceMeta<M::Provenance>,
        pointee: TyAndLayout<'tcx>,
    ) -> InterpResult<'tcx> {
        let tail = self.ecx.tcx.struct_tail_erasing_lifetimes(pointee.ty, self.ecx.param_env);
        match tail.kind() {
            ty::Dynamic(data, _, ty::Dyn) => {
                let vtable = meta.unwrap_meta().to_pointer(self.ecx)?;
                // Make sure it is a genuine vtable pointer for the right trait.
                try_validation!(
                    self.ecx.get_ptr_vtable_ty(vtable, Some(data)),
                    self.path,
                    Ub(DanglingIntPointer(..) | InvalidVTablePointer(..)) =>
                        InvalidVTablePtr { value: format!("{vtable}") },
                    Ub(InvalidVTableTrait { expected_trait, vtable_trait }) => {
                        InvalidMetaWrongTrait { expected_trait, vtable_trait: *vtable_trait }
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

        Ok(())
    }

    /// Check a reference or `Box`.
    fn check_safe_pointer(
        &mut self,
        value: &OpTy<'tcx, M::Provenance>,
        ptr_kind: PointerKind,
    ) -> InterpResult<'tcx> {
        // Not using `deref_pointer` since we want to use our `read_immediate` wrapper.
        let place = self.ecx.ref_to_mplace(&self.read_immediate(value, ptr_kind.into())?)?;
        // Handle wide pointers.
        // Check metadata early, for better diagnostics
        if place.layout.is_unsized() {
            self.check_wide_ptr_meta(place.meta(), place.layout)?;
        }
        // Make sure this is dereferenceable and all.
        let size_and_align = try_validation!(
            self.ecx.size_and_align_of_mplace(&place),
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
                CheckInAllocMsg::InboundsTest, // will anyway be replaced by validity message
            ),
            self.path,
            Ub(DanglingIntPointer(0, _)) => NullPtr { ptr_kind },
            Ub(DanglingIntPointer(i, _)) => DanglingPtrNoProvenance {
                ptr_kind,
                // FIXME this says "null pointer" when null but we need translate
                pointer: format!("{}", Pointer::<Option<AllocId>>::from_addr_invalid(*i))
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
        // Do not allow pointers to uninhabited types.
        if place.layout.abi.is_uninhabited() {
            let ty = place.layout.ty;
            throw_validation_failure!(self.path, PtrToUninhabited { ptr_kind, ty })
        }
        // Recursive checking
        if let Some(ref_tracking) = self.ref_tracking.as_deref_mut() {
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
            // Proceed recursively even for ZST, no reason to skip them!
            // `!` is a ZST and we want to validate it.
            if let Ok((alloc_id, _offset, _prov)) = self.ecx.ptr_try_get_alloc_id(place.ptr()) {
                let mut skip_recursive_check = false;
                if let Some(GlobalAlloc::Static(did)) = self.ecx.tcx.try_get_global_alloc(alloc_id)
                {
                    let DefKind::Static { nested, .. } = self.ecx.tcx.def_kind(did) else { bug!() };
                    // Special handling for pointers to statics (irrespective of their type).
                    assert!(!self.ecx.tcx.is_thread_local_static(did));
                    assert!(self.ecx.tcx.is_static(did));
                    // Mode-specific checks
                    match self.ctfe_mode {
                        Some(
                            CtfeValidationMode::Static { .. } | CtfeValidationMode::Promoted { .. },
                        ) => {
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
                        Some(CtfeValidationMode::Const { .. }) => {
                            // We can't recursively validate `extern static`, so we better reject them.
                            if self.ecx.tcx.is_foreign_item(did) {
                                throw_validation_failure!(self.path, ConstRefToExtern);
                            }
                        }
                        None => {}
                    }
                }

                // Dangling and Mutability check.
                let (size, _align, alloc_kind) = self.ecx.get_alloc_info(alloc_id);
                if alloc_kind == AllocKind::Dead {
                    // This can happen for zero-sized references. We can't have *any* references to non-existing
                    // allocations though, interning rejects them all as the rest of rustc isn't happy with them...
                    // so we throw an error, even though this isn't really UB.
                    // A potential future alternative would be to resurrect this as a zero-sized allocation
                    // (which codegen will then compile to an aligned dummy pointer anyway).
                    throw_validation_failure!(self.path, DanglingPtrUseAfterFree { ptr_kind });
                }
                // If this allocation has size zero, there is no actual mutability here.
                if size != Size::ZERO {
                    let alloc_actual_mutbl = mutability(self.ecx, alloc_id);
                    // Mutable pointer to immutable memory is no good.
                    if ptr_expected_mutbl == Mutability::Mut
                        && alloc_actual_mutbl == Mutability::Not
                    {
                        throw_validation_failure!(self.path, MutableRefToImmutable);
                    }
                    // In a const, everything must be completely immutable.
                    if matches!(self.ctfe_mode, Some(CtfeValidationMode::Const { .. })) {
                        if ptr_expected_mutbl == Mutability::Mut
                            || alloc_actual_mutbl == Mutability::Mut
                        {
                            throw_validation_failure!(self.path, ConstRefToMutable);
                        }
                    }
                }
                // Potentially skip recursive check.
                if skip_recursive_check {
                    return Ok(());
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
        Ok(())
    }

    /// Check if this is a value of primitive type, and if yes check the validity of the value
    /// at that type. Return `true` if the type is indeed primitive.
    ///
    /// Note that not all of these have `FieldsShape::Primitive`, e.g. wide references.
    fn try_visit_primitive(
        &mut self,
        value: &OpTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, bool> {
        // Go over all the primitive types
        let ty = value.layout.ty;
        match ty.kind() {
            ty::Bool => {
                let value = self.read_scalar(value, ExpectedKind::Bool)?;
                try_validation!(
                    value.to_bool(),
                    self.path,
                    Ub(InvalidBool(..)) => ValidationErrorKind::InvalidBool {
                        value: format!("{value:x}"),
                    }
                );
                Ok(true)
            }
            ty::Char => {
                let value = self.read_scalar(value, ExpectedKind::Char)?;
                try_validation!(
                    value.to_char(),
                    self.path,
                    Ub(InvalidChar(..)) => ValidationErrorKind::InvalidChar {
                        value: format!("{value:x}"),
                    }
                );
                Ok(true)
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
                Ok(true)
            }
            ty::RawPtr(..) => {
                let place =
                    self.ecx.ref_to_mplace(&self.read_immediate(value, ExpectedKind::RawPtr)?)?;
                if place.layout.is_unsized() {
                    self.check_wide_ptr_meta(place.meta(), place.layout)?;
                }
                Ok(true)
            }
            ty::Ref(_, _ty, mutbl) => {
                self.check_safe_pointer(value, PointerKind::Ref(*mutbl))?;
                Ok(true)
            }
            ty::FnPtr(_sig) => {
                let value = self.read_scalar(value, ExpectedKind::FnPtr)?;

                // If we check references recursively, also check that this points to a function.
                if let Some(_) = self.ref_tracking {
                    let ptr = value.to_pointer(self.ecx)?;
                    let _fn = try_validation!(
                        self.ecx.get_ptr_fn(ptr),
                        self.path,
                        Ub(DanglingIntPointer(..) | InvalidFunctionPointer(..)) =>
                            InvalidFnPtr { value: format!("{ptr}") },
                    );
                    // FIXME: Check if the signature matches
                } else {
                    // Otherwise (for standalone Miri), we have to still check it to be non-null.
                    if self.ecx.scalar_may_be_null(value)? {
                        throw_validation_failure!(self.path, NullFnPtr);
                    }
                }
                Ok(true)
            }
            ty::Never => throw_validation_failure!(self.path, NeverVal),
            ty::Foreign(..) | ty::FnDef(..) => {
                // Nothing to check.
                Ok(true)
            }
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
            | ty::Coroutine(..) => Ok(false),
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
                        return Ok(());
                    }
                } else if scalar_layout.is_always_valid(self.ecx) {
                    // Easy. (This is reachable if `enforce_number_validity` is set.)
                    return Ok(());
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
            Ok(())
        } else {
            throw_validation_failure!(
                self.path,
                OutOfRange { value: format!("{bits}"), range: valid_range, max_value }
            )
        }
    }

    fn in_mutable_memory(&self, op: &OpTy<'tcx, M::Provenance>) -> bool {
        if let Some(mplace) = op.as_mplace_or_imm().left() {
            if let Some(alloc_id) = mplace.ptr().provenance.and_then(|p| p.get_alloc_id()) {
                return mutability(self.ecx, alloc_id).is_mut();
            }
        }
        false
    }
}

/// Returns whether the allocation is mutable, and whether it's actually a static.
/// For "root" statics we look at the type to account for interior
/// mutability; for nested statics we have no type and directly use the annotated mutability.
fn mutability<'tcx>(ecx: &InterpCx<'tcx, impl Machine<'tcx>>, alloc_id: AllocId) -> Mutability {
    // Let's see what kind of memory this points to.
    // We're not using `try_global_alloc` since dangling pointers have already been handled.
    match ecx.tcx.global_alloc(alloc_id) {
        GlobalAlloc::Static(did) => {
            let DefKind::Static { safety: _, mutability, nested } = ecx.tcx.def_kind(did) else {
                bug!()
            };
            if nested {
                assert!(
                    ecx.memory.alloc_map.get(alloc_id).is_none(),
                    "allocations of nested statics are already interned: {alloc_id:?}, {did:?}"
                );
                // Nested statics in a `static` are never interior mutable,
                // so just use the declared mutability.
                mutability
            } else {
                let mutability = match mutability {
                    Mutability::Not
                        if !ecx
                            .tcx
                            .type_of(did)
                            .no_bound_vars()
                            .expect("statics should not have generic parameters")
                            .is_freeze(*ecx.tcx, ty::ParamEnv::reveal_all()) =>
                    {
                        Mutability::Mut
                    }
                    _ => mutability,
                };
                if let Some((_, alloc)) = ecx.memory.alloc_map.get(alloc_id) {
                    assert_eq!(alloc.mutability, mutability);
                }
                mutability
            }
        }
        GlobalAlloc::Memory(alloc) => alloc.inner().mutability,
        GlobalAlloc::Function { .. } | GlobalAlloc::VTable(..) => {
            // These are immutable, we better don't allow mutable pointers here.
            Mutability::Not
        }
    }
}

impl<'rt, 'tcx, M: Machine<'tcx>> ValueVisitor<'tcx, M> for ValidityVisitor<'rt, 'tcx, M> {
    type V = OpTy<'tcx, M::Provenance>;

    #[inline(always)]
    fn ecx(&self) -> &InterpCx<'tcx, M> {
        self.ecx
    }

    fn read_discriminant(
        &mut self,
        op: &OpTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, VariantIdx> {
        self.with_elem(PathElem::EnumTag, move |this| {
            Ok(try_validation!(
                this.ecx.read_discriminant(op),
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
        old_op: &OpTy<'tcx, M::Provenance>,
        field: usize,
        new_op: &OpTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx> {
        let elem = self.aggregate_field_path_elem(old_op.layout, field);
        self.with_elem(elem, move |this| this.visit_value(new_op))
    }

    #[inline]
    fn visit_variant(
        &mut self,
        old_op: &OpTy<'tcx, M::Provenance>,
        variant_id: VariantIdx,
        new_op: &OpTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx> {
        let name = match old_op.layout.ty.kind() {
            ty::Adt(adt, _) => PathElem::Variant(adt.variant(variant_id).name),
            // Coroutines also have variants
            ty::Coroutine(..) => PathElem::CoroutineState(variant_id),
            _ => bug!("Unexpected type with variant: {:?}", old_op.layout.ty),
        };
        self.with_elem(name, move |this| this.visit_value(new_op))
    }

    #[inline(always)]
    fn visit_union(
        &mut self,
        op: &OpTy<'tcx, M::Provenance>,
        _fields: NonZero<usize>,
    ) -> InterpResult<'tcx> {
        // Special check for CTFE validation, preventing `UnsafeCell` inside unions in immutable memory.
        if self.ctfe_mode.is_some_and(|c| !c.allow_immutable_unsafe_cell()) {
            if !op.layout.is_zst() && !op.layout.ty.is_freeze(*self.ecx.tcx, self.ecx.param_env) {
                if !self.in_mutable_memory(op) {
                    throw_validation_failure!(self.path, UnsafeCellInImmutable);
                }
            }
        }
        Ok(())
    }

    #[inline]
    fn visit_box(
        &mut self,
        _box_ty: Ty<'tcx>,
        op: &OpTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx> {
        self.check_safe_pointer(op, PointerKind::Box)?;
        Ok(())
    }

    #[inline]
    fn visit_value(&mut self, op: &OpTy<'tcx, M::Provenance>) -> InterpResult<'tcx> {
        trace!("visit_value: {:?}, {:?}", *op, op.layout);

        // Check primitive types -- the leaves of our recursive descent.
        // We assume that the Scalar validity range does not restrict these values
        // any further than `try_visit_primitive` does!
        if self.try_visit_primitive(op)? {
            return Ok(());
        }

        // Special check preventing `UnsafeCell` in the inner part of constants
        if self.ctfe_mode.is_some_and(|c| !c.allow_immutable_unsafe_cell()) {
            if !op.layout.is_zst()
                && let Some(def) = op.layout.ty.ty_adt_def()
                && def.is_unsafe_cell()
            {
                if !self.in_mutable_memory(op) {
                    throw_validation_failure!(self.path, UnsafeCellInImmutable);
                }
            }
        }

        // Recursively walk the value at its type. Apply optimizations for some large types.
        match op.layout.ty.kind() {
            ty::Str => {
                let mplace = op.assert_mem_place(); // strings are unsized and hence never immediate
                let len = mplace.len(self.ecx)?;
                try_validation!(
                    self.ecx.read_bytes_ptr_strip_provenance(mplace.ptr(), Size::from_bytes(len)),
                    self.path,
                    Ub(InvalidUninitBytes(..)) => Uninit { expected: ExpectedKind::Str },
                    Unsup(ReadPointerAsInt(_)) => PointerAsInt { expected: ExpectedKind::Str }
                );
            }
            ty::Array(tys, ..) | ty::Slice(tys)
                // This optimization applies for types that can hold arbitrary bytes (such as
                // integer and floating point types) or for structs or tuples with no fields.
                // FIXME(wesleywiser) This logic could be extended further to arbitrary structs
                // or tuples made up of integer/floating point types or inhabited ZSTs with no
                // padding.
                if matches!(tys.kind(), ty::Int(..) | ty::Uint(..) | ty::Float(..))
                =>
            {
                let expected = if tys.is_integral() { ExpectedKind::Int } else { ExpectedKind::Float };
                // Optimized handling for arrays of integer/float type.

                // This is the length of the array/slice.
                let len = op.len(self.ecx)?;
                // This is the element type size.
                let layout = self.ecx.layout_of(*tys)?;
                // This is the size in bytes of the whole array. (This checks for overflow.)
                let size = layout.size * len;
                // If the size is 0, there is nothing to check.
                // (`size` can only be 0 of `len` is 0, and empty arrays are always valid.)
                if size == Size::ZERO {
                    return Ok(());
                }
                // Now that we definitely have a non-ZST array, we know it lives in memory.
                let mplace = match op.as_mplace_or_imm() {
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

                match alloc.get_bytes_strip_provenance() {
                    // In the happy case, we needn't check anything else.
                    Ok(_) => {}
                    // Some error happened, try to provide a more detailed description.
                    Err(err) => {
                        // For some errors we might be able to provide extra information.
                        // (This custom logic does not fit the `try_validation!` macro.)
                        match err.kind() {
                            Ub(InvalidUninitBytes(Some((_alloc_id, access)))) | Unsup(ReadPointerAsInt(Some((_alloc_id, access)))) => {
                                // Some byte was uninitialized, determine which
                                // element that byte belongs to so we can
                                // provide an index.
                                let i = usize::try_from(
                                    access.bad.start.bytes() / layout.size.bytes(),
                                )
                                .unwrap();
                                self.path.push(PathElem::ArrayElem(i));

                                if matches!(err.kind(), Ub(InvalidUninitBytes(_))) {
                                    throw_validation_failure!(self.path, Uninit { expected })
                                } else {
                                    throw_validation_failure!(self.path, PointerAsInt { expected })
                                }
                            }

                            // Propagate upwards (that will also check for unexpected errors).
                            _ => return Err(err),
                        }
                    }
                }
            }
            // Fast path for arrays and slices of ZSTs. We only need to check a single ZST element
            // of an array and not all of them, because there's only a single value of a specific
            // ZST type, so either validation fails for all elements or none.
            ty::Array(tys, ..) | ty::Slice(tys) if self.ecx.layout_of(*tys)?.is_zst() => {
                // Validate just the first element (if any).
                if op.len(self.ecx)? > 0 {
                    self.visit_field(op, 0, &self.ecx.project_index(op, 0)?)?;
                }
            }
            _ => {
                // default handler
                try_validation!(
                    self.walk_value(op),
                    self.path,
                    // It's not great to catch errors here, since we can't give a very good path,
                    // but it's better than ICEing.
                    Ub(InvalidVTableTrait { expected_trait, vtable_trait }) => {
                        InvalidMetaWrongTrait { expected_trait, vtable_trait: *vtable_trait }
                    },
                );
            }
        }

        // *After* all of this, check the ABI. We need to check the ABI to handle
        // types like `NonNull` where the `Scalar` info is more restrictive than what
        // the fields say (`rustc_layout_scalar_valid_range_start`).
        // But in most cases, this will just propagate what the fields say,
        // and then we want the error to point at the field -- so, first recurse,
        // then check ABI.
        //
        // FIXME: We could avoid some redundant checks here. For newtypes wrapping
        // scalars, we do the same check on every "level" (e.g., first we check
        // MyNewtype and then the scalar in there).
        match op.layout.abi {
            Abi::Uninhabited => {
                let ty = op.layout.ty;
                throw_validation_failure!(self.path, UninhabitedVal { ty });
            }
            Abi::Scalar(scalar_layout) => {
                if !scalar_layout.is_uninit_valid() {
                    // There is something to check here.
                    let scalar = self.read_scalar(op, ExpectedKind::InitScalar)?;
                    self.visit_scalar(scalar, scalar_layout)?;
                }
            }
            Abi::ScalarPair(a_layout, b_layout) => {
                // We can only proceed if *both* scalars need to be initialized.
                // FIXME: find a way to also check ScalarPair when one side can be uninit but
                // the other must be init.
                if !a_layout.is_uninit_valid() && !b_layout.is_uninit_valid() {
                    let (a, b) =
                        self.read_immediate(op, ExpectedKind::InitScalar)?.to_scalar_pair();
                    self.visit_scalar(a, a_layout)?;
                    self.visit_scalar(b, b_layout)?;
                }
            }
            Abi::Vector { .. } => {
                // No checks here, we assume layout computation gets this right.
                // (This is harder to check since Miri does not represent these as `Immediate`. We
                // also cannot use field projections since this might be a newtype around a vector.)
            }
            Abi::Aggregate { .. } => {
                // Nothing to do.
            }
        }

        Ok(())
    }
}

impl<'tcx, M: Machine<'tcx>> InterpCx<'tcx, M> {
    fn validate_operand_internal(
        &self,
        op: &OpTy<'tcx, M::Provenance>,
        path: Vec<PathElem>,
        ref_tracking: Option<&mut RefTracking<MPlaceTy<'tcx, M::Provenance>, Vec<PathElem>>>,
        ctfe_mode: Option<CtfeValidationMode>,
    ) -> InterpResult<'tcx> {
        trace!("validate_operand_internal: {:?}, {:?}", *op, op.layout.ty);

        // Construct a visitor
        let mut visitor = ValidityVisitor { path, ref_tracking, ctfe_mode, ecx: self };

        // Run it.
        match self.run_for_validation(|| visitor.visit_value(op)) {
            Ok(()) => Ok(()),
            // Pass through validation failures and "invalid program" issues.
            Err(err)
                if matches!(
                    err.kind(),
                    err_ub!(ValidationError { .. })
                        | InterpError::InvalidProgram(_)
                        | InterpError::Unsupported(UnsupportedOpInfo::ExternTypeField)
                ) =>
            {
                Err(err)
            }
            // Complain about any other kind of error -- those are bad because we'd like to
            // report them in a way that shows *where* in the value the issue lies.
            Err(err) => {
                bug!(
                    "Unexpected error during validation: {}",
                    format_interp_error(self.tcx.dcx(), err)
                );
            }
        }
    }

    /// This function checks the data at `op` to be const-valid.
    /// `op` is assumed to cover valid memory if it is an indirect operand.
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
        &self,
        op: &OpTy<'tcx, M::Provenance>,
        path: Vec<PathElem>,
        ref_tracking: &mut RefTracking<MPlaceTy<'tcx, M::Provenance>, Vec<PathElem>>,
        ctfe_mode: CtfeValidationMode,
    ) -> InterpResult<'tcx> {
        self.validate_operand_internal(op, path, Some(ref_tracking), Some(ctfe_mode))
    }

    /// This function checks the data at `op` to be runtime-valid.
    /// `op` is assumed to cover valid memory if it is an indirect operand.
    /// It will error if the bits at the destination do not match the ones described by the layout.
    #[inline(always)]
    pub fn validate_operand(&self, op: &OpTy<'tcx, M::Provenance>) -> InterpResult<'tcx> {
        // Note that we *could* actually be in CTFE here with `-Zextra-const-ub-checks`, but it's
        // still correct to not use `ctfe_mode`: that mode is for validation of the final constant
        // value, it rules out things like `UnsafeCell` in awkward places. It also can make checking
        // recurse through references which, for now, we don't want here, either.
        self.validate_operand_internal(op, vec![], None, None)
    }
}
