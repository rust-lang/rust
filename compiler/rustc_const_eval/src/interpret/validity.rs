//! Check the validity invariant of a given value, and tell the user
//! where in the value it got violated.
//! In const context, this goes even further and tries to approximate const safety.
//! That's useful because it means other passes (e.g. promotion) can rely on `const`s
//! to be const-safe.

use std::convert::TryFrom;
use std::fmt::Write;
use std::num::NonZeroUsize;

use rustc_ast::Mutability;
use rustc_data_structures::fx::FxHashSet;
use rustc_hir as hir;
use rustc_middle::mir::interpret::InterpError;
use rustc_middle::ty;
use rustc_middle::ty::layout::{LayoutOf, TyAndLayout};
use rustc_span::symbol::{sym, Symbol};
use rustc_span::DUMMY_SP;
use rustc_target::abi::{Abi, Scalar as ScalarAbi, Size, VariantIdx, Variants, WrappingRange};

use std::hash::Hash;

// for the validation errors
use super::UndefinedBehaviorInfo::*;
use super::{
    CheckInAllocMsg, GlobalAlloc, ImmTy, Immediate, InterpCx, InterpResult, MPlaceTy, Machine,
    MemPlaceMeta, OpTy, Scalar, ValueVisitor,
};

macro_rules! throw_validation_failure {
    ($where:expr, { $( $what_fmt:expr ),+ } $( expected { $( $expected_fmt:expr ),+ } )?) => {{
        let mut msg = String::new();
        msg.push_str("encountered ");
        write!(&mut msg, $($what_fmt),+).unwrap();
        $(
            msg.push_str(", but expected ");
            write!(&mut msg, $($expected_fmt),+).unwrap();
        )?
        let path = rustc_middle::ty::print::with_no_trimmed_paths!({
            let where_ = &$where;
            if !where_.is_empty() {
                let mut path = String::new();
                write_path(&mut path, where_);
                Some(path)
            } else {
                None
            }
        });
        throw_ub!(ValidationFailure { path, msg })
    }};
}

/// If $e throws an error matching the pattern, throw a validation failure.
/// Other errors are passed back to the caller, unchanged -- and if they reach the root of
/// the visitor, we make sure only validation errors and `InvalidProgram` errors are left.
/// This lets you use the patterns as a kind of validation list, asserting which errors
/// can possibly happen:
///
/// ```
/// let v = try_validation!(some_fn(), some_path, {
///     Foo | Bar | Baz => { "some failure" },
/// });
/// ```
///
/// The patterns must be of type `UndefinedBehaviorInfo`.
/// An additional expected parameter can also be added to the failure message:
///
/// ```
/// let v = try_validation!(some_fn(), some_path, {
///     Foo | Bar | Baz => { "some failure" } expected { "something that wasn't a failure" },
/// });
/// ```
///
/// An additional nicety is that both parameters actually take format args, so you can just write
/// the format string in directly:
///
/// ```
/// let v = try_validation!(some_fn(), some_path, {
///     Foo | Bar | Baz => { "{:?}", some_failure } expected { "{}", expected_value },
/// });
/// ```
///
macro_rules! try_validation {
    ($e:expr, $where:expr,
    $( $( $p:pat_param )|+ => { $( $what_fmt:expr ),+ } $( expected { $( $expected_fmt:expr ),+ } )? ),+ $(,)?
    ) => {{
        match $e {
            Ok(x) => x,
            // We catch the error and turn it into a validation failure. We are okay with
            // allocation here as this can only slow down builds that fail anyway.
            Err(e) => match e.kind() {
                $(
                    InterpError::UndefinedBehavior($($p)|+) =>
                       throw_validation_failure!(
                            $where,
                            { $( $what_fmt ),+ } $( expected { $( $expected_fmt ),+ } )?
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
    GeneratorState(VariantIdx),
    CapturedVar(Symbol),
    ArrayElem(usize),
    TupleElem(usize),
    Deref,
    EnumTag,
    GeneratorTag,
    DynDowncast,
}

/// Extra things to check for during validation of CTFE results.
pub enum CtfeValidationMode {
    /// Regular validation, nothing special happening.
    Regular,
    /// Validation of a `const`.
    /// `inner` says if this is an inner, indirect allocation (as opposed to the top-level const
    /// allocation). Being an inner allocation makes a difference because the top-level allocation
    /// of a `const` is copied for each use, but the inner allocations are implicitly shared.
    /// `allow_static_ptrs` says if pointers to statics are permitted (which is the case for promoteds in statics).
    Const { inner: bool, allow_static_ptrs: bool },
}

/// State for tracking recursive validation of references
pub struct RefTracking<T, PATH = ()> {
    pub seen: FxHashSet<T>,
    pub todo: Vec<(T, PATH)>,
}

impl<T: Copy + Eq + Hash + std::fmt::Debug, PATH: Default> RefTracking<T, PATH> {
    pub fn empty() -> Self {
        RefTracking { seen: FxHashSet::default(), todo: vec![] }
    }
    pub fn new(op: T) -> Self {
        let mut ref_tracking_for_consts =
            RefTracking { seen: FxHashSet::default(), todo: vec![(op, PATH::default())] };
        ref_tracking_for_consts.seen.insert(op);
        ref_tracking_for_consts
    }

    pub fn track(&mut self, op: T, path: impl FnOnce() -> PATH) {
        if self.seen.insert(op) {
            trace!("Recursing below ptr {:#?}", op);
            let path = path();
            // Remember to come back to this later.
            self.todo.push((op, path));
        }
    }
}

/// Format a path
fn write_path(out: &mut String, path: &[PathElem]) {
    use self::PathElem::*;

    for elem in path.iter() {
        match elem {
            Field(name) => write!(out, ".{}", name),
            EnumTag => write!(out, ".<enum-tag>"),
            Variant(name) => write!(out, ".<enum-variant({})>", name),
            GeneratorTag => write!(out, ".<generator-tag>"),
            GeneratorState(idx) => write!(out, ".<generator-state({})>", idx.index()),
            CapturedVar(name) => write!(out, ".<captured-var({})>", name),
            TupleElem(idx) => write!(out, ".{}", idx),
            ArrayElem(idx) => write!(out, "[{}]", idx),
            // `.<deref>` does not match Rust syntax, but it is more readable for long paths -- and
            // some of the other items here also are not Rust syntax.  Actually we can't
            // even use the usual syntax because we are just showing the projections,
            // not the root.
            Deref => write!(out, ".<deref>"),
            DynDowncast => write!(out, ".<dyn-downcast>"),
        }
        .unwrap()
    }
}

// Formats such that a sentence like "expected something {}" to mean
// "expected something <in the given range>" makes sense.
fn wrapping_range_format(r: WrappingRange, max_hi: u128) -> String {
    let WrappingRange { start: lo, end: hi } = r;
    assert!(hi <= max_hi);
    if lo > hi {
        format!("less or equal to {}, or greater or equal to {}", hi, lo)
    } else if lo == hi {
        format!("equal to {}", lo)
    } else if lo == 0 {
        assert!(hi < max_hi, "should not be printing if the range covers everything");
        format!("less or equal to {}", hi)
    } else if hi == max_hi {
        assert!(lo > 0, "should not be printing if the range covers everything");
        format!("greater or equal to {}", lo)
    } else {
        format!("in the range {:?}", r)
    }
}

struct ValidityVisitor<'rt, 'mir, 'tcx, M: Machine<'mir, 'tcx>> {
    /// The `path` may be pushed to, but the part that is present when a function
    /// starts must not be changed!  `visit_fields` and `visit_array` rely on
    /// this stack discipline.
    path: Vec<PathElem>,
    ref_tracking: Option<&'rt mut RefTracking<MPlaceTy<'tcx, M::Provenance>, Vec<PathElem>>>,
    /// `None` indicates this is not validating for CTFE (but for runtime).
    ctfe_mode: Option<CtfeValidationMode>,
    ecx: &'rt InterpCx<'mir, 'tcx, M>,
}

impl<'rt, 'mir, 'tcx: 'mir, M: Machine<'mir, 'tcx>> ValidityVisitor<'rt, 'mir, 'tcx, M> {
    fn aggregate_field_path_elem(&mut self, layout: TyAndLayout<'tcx>, field: usize) -> PathElem {
        // First, check if we are projecting to a variant.
        match layout.variants {
            Variants::Multiple { tag_field, .. } => {
                if tag_field == field {
                    return match layout.ty.kind() {
                        ty::Adt(def, ..) if def.is_enum() => PathElem::EnumTag,
                        ty::Generator(..) => PathElem::GeneratorTag,
                        _ => bug!("non-variant type {:?}", layout.ty),
                    };
                }
            }
            Variants::Single { .. } => {}
        }

        // Now we know we are projecting to a field, so figure out which one.
        match layout.ty.kind() {
            // generators and closures.
            ty::Closure(def_id, _) | ty::Generator(def_id, _, _) => {
                let mut name = None;
                // FIXME this should be more descriptive i.e. CapturePlace instead of CapturedVar
                // https://github.com/rust-lang/project-rfc-2229/issues/46
                if let Some(local_def_id) = def_id.as_local() {
                    let tables = self.ecx.tcx.typeck(local_def_id);
                    if let Some(captured_place) =
                        tables.closure_min_captures_flattened(local_def_id).nth(field)
                    {
                        // Sometimes the index is beyond the number of upvars (seen
                        // for a generator).
                        let var_hir_id = captured_place.get_root_variable();
                        let node = self.ecx.tcx.hir().get(var_hir_id);
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
                        PathElem::Field(def.variant(index).fields[field].name)
                    }
                    Variants::Multiple { .. } => bug!("we handled variants above"),
                }
            }

            // other ADTs
            ty::Adt(def, _) => PathElem::Field(def.non_enum_variant().fields[field].name),

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
        expected: &str,
    ) -> InterpResult<'tcx, ImmTy<'tcx, M::Provenance>> {
        Ok(try_validation!(
            self.ecx.read_immediate(op),
            self.path,
            InvalidUninitBytes(None) => { "uninitialized memory" } expected { "{expected}" }
        ))
    }

    fn read_scalar(
        &self,
        op: &OpTy<'tcx, M::Provenance>,
        expected: &str,
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
            ty::Dynamic(..) => {
                let vtable = meta.unwrap_meta().to_pointer(self.ecx)?;
                // Make sure it is a genuine vtable pointer.
                let (_ty, _trait) = try_validation!(
                    self.ecx.get_ptr_vtable(vtable),
                    self.path,
                    DanglingIntPointer(..) |
                    InvalidVTablePointer(..) =>
                        { "{vtable}" } expected { "a vtable pointer" },
                );
                // FIXME: check if the type/trait match what ty::Dynamic says?
            }
            ty::Slice(..) | ty::Str => {
                let _len = meta.unwrap_meta().to_machine_usize(self.ecx)?;
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
        kind: &str,
    ) -> InterpResult<'tcx> {
        let place = self.ecx.ref_to_mplace(&self.read_immediate(value, &format!("a {kind}"))?)?;
        // Handle wide pointers.
        // Check metadata early, for better diagnostics
        if place.layout.is_unsized() {
            self.check_wide_ptr_meta(place.meta, place.layout)?;
        }
        // Make sure this is dereferenceable and all.
        let size_and_align = try_validation!(
            self.ecx.size_and_align_of_mplace(&place),
            self.path,
            InvalidMeta(msg) => { "invalid {} metadata: {}", kind, msg },
        );
        let (size, align) = size_and_align
            // for the purpose of validity, consider foreign types to have
            // alignment and size determined by the layout (size will be 0,
            // alignment should take attributes into account).
            .unwrap_or_else(|| (place.layout.size, place.layout.align.abi));
        // Direct call to `check_ptr_access_align` checks alignment even on CTFE machines.
        try_validation!(
            self.ecx.check_ptr_access_align(
                place.ptr,
                size,
                align,
                CheckInAllocMsg::InboundsTest, // will anyway be replaced by validity message
            ),
            self.path,
            AlignmentCheckFailed { required, has } =>
                {
                    "an unaligned {kind} (required {} byte alignment but found {})",
                    required.bytes(),
                    has.bytes()
                },
            DanglingIntPointer(0, _) =>
                { "a null {kind}" },
            DanglingIntPointer(i, _) =>
                { "a dangling {kind} (address {i:#x} is unallocated)" },
            PointerOutOfBounds { .. } =>
                { "a dangling {kind} (going beyond the bounds of its allocation)" },
            // This cannot happen during const-eval (because interning already detects
            // dangling pointers), but it can happen in Miri.
            PointerUseAfterFree(..) =>
                { "a dangling {kind} (use-after-free)" },
        );
        // Do not allow pointers to uninhabited types.
        if place.layout.abi.is_uninhabited() {
            throw_validation_failure!(self.path,
                { "a {kind} pointing to uninhabited type {}", place.layout.ty }
            )
        }
        // Recursive checking
        if let Some(ref mut ref_tracking) = self.ref_tracking {
            // Proceed recursively even for ZST, no reason to skip them!
            // `!` is a ZST and we want to validate it.
            if let Ok((alloc_id, _offset, _prov)) = self.ecx.ptr_try_get_alloc_id(place.ptr) {
                // Let's see what kind of memory this points to.
                let alloc_kind = self.ecx.tcx.try_get_global_alloc(alloc_id);
                match alloc_kind {
                    Some(GlobalAlloc::Static(did)) => {
                        // Special handling for pointers to statics (irrespective of their type).
                        assert!(!self.ecx.tcx.is_thread_local_static(did));
                        assert!(self.ecx.tcx.is_static(did));
                        if matches!(
                            self.ctfe_mode,
                            Some(CtfeValidationMode::Const { allow_static_ptrs: false, .. })
                        ) {
                            // See const_eval::machine::MemoryExtra::can_access_statics for why
                            // this check is so important.
                            // This check is reachable when the const just referenced the static,
                            // but never read it (so we never entered `before_access_global`).
                            throw_validation_failure!(self.path,
                                { "a {} pointing to a static variable in a constant", kind }
                            );
                        }
                        // We skip recursively checking other statics. These statics must be sound by
                        // themselves, and the only way to get broken statics here is by using
                        // unsafe code.
                        // The reasons we don't check other statics is twofold. For one, in all
                        // sound cases, the static was already validated on its own, and second, we
                        // trigger cycle errors if we try to compute the value of the other static
                        // and that static refers back to us.
                        // We might miss const-invalid data,
                        // but things are still sound otherwise (in particular re: consts
                        // referring to statics).
                        return Ok(());
                    }
                    Some(GlobalAlloc::Memory(alloc)) => {
                        if alloc.inner().mutability == Mutability::Mut
                            && matches!(self.ctfe_mode, Some(CtfeValidationMode::Const { .. }))
                        {
                            // This should be unreachable, but if someone manages to copy a pointer
                            // out of a `static`, then that pointer might point to mutable memory,
                            // and we would catch that here.
                            throw_validation_failure!(self.path,
                                { "a {} pointing to mutable memory in a constant", kind }
                            );
                        }
                    }
                    // Nothing to check for these.
                    None | Some(GlobalAlloc::Function(..) | GlobalAlloc::VTable(..)) => {}
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
    /// at that type.  Return `true` if the type is indeed primitive.
    fn try_visit_primitive(
        &mut self,
        value: &OpTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, bool> {
        // Go over all the primitive types
        let ty = value.layout.ty;
        match ty.kind() {
            ty::Bool => {
                let value = self.read_scalar(value, "a boolean")?;
                try_validation!(
                    value.to_bool(),
                    self.path,
                    InvalidBool(..) =>
                        { "{:x}", value } expected { "a boolean" },
                );
                Ok(true)
            }
            ty::Char => {
                let value = self.read_scalar(value, "a unicode scalar value")?;
                try_validation!(
                    value.to_char(),
                    self.path,
                    InvalidChar(..) =>
                        { "{:x}", value } expected { "a valid unicode scalar value (in `0..=0x10FFFF` but not in `0xD800..=0xDFFF`)" },
                );
                Ok(true)
            }
            ty::Float(_) | ty::Int(_) | ty::Uint(_) => {
                // NOTE: Keep this in sync with the array optimization for int/float
                // types below!
                let value = self.read_scalar(
                    value,
                    if matches!(ty.kind(), ty::Float(..)) {
                        "a floating point number"
                    } else {
                        "an integer"
                    },
                )?;
                // As a special exception we *do* match on a `Scalar` here, since we truly want
                // to know its underlying representation (and *not* cast it to an integer).
                if matches!(value, Scalar::Ptr(..)) {
                    throw_validation_failure!(self.path,
                        { "{:x}", value } expected { "plain (non-pointer) bytes" }
                    )
                }
                Ok(true)
            }
            ty::RawPtr(..) => {
                // We are conservative with uninit for integers, but try to
                // actually enforce the strict rules for raw pointers (mostly because
                // that lets us re-use `ref_to_mplace`).
                let place =
                    self.ecx.ref_to_mplace(&self.read_immediate(value, "a raw pointer")?)?;
                if place.layout.is_unsized() {
                    self.check_wide_ptr_meta(place.meta, place.layout)?;
                }
                Ok(true)
            }
            ty::Ref(_, ty, mutbl) => {
                if matches!(self.ctfe_mode, Some(CtfeValidationMode::Const { .. }))
                    && *mutbl == Mutability::Mut
                {
                    // A mutable reference inside a const? That does not seem right (except if it is
                    // a ZST).
                    let layout = self.ecx.layout_of(*ty)?;
                    if !layout.is_zst() {
                        throw_validation_failure!(self.path, { "mutable reference in a `const`" });
                    }
                }
                self.check_safe_pointer(value, "reference")?;
                Ok(true)
            }
            ty::FnPtr(_sig) => {
                let value = self.read_scalar(value, "a function pointer")?;

                // If we check references recursively, also check that this points to a function.
                if let Some(_) = self.ref_tracking {
                    let ptr = value.to_pointer(self.ecx)?;
                    let _fn = try_validation!(
                        self.ecx.get_ptr_fn(ptr),
                        self.path,
                        DanglingIntPointer(..) |
                        InvalidFunctionPointer(..) =>
                            { "{ptr}" } expected { "a function pointer" },
                    );
                    // FIXME: Check if the signature matches
                } else {
                    // Otherwise (for standalone Miri), we have to still check it to be non-null.
                    if self.ecx.scalar_may_be_null(value)? {
                        throw_validation_failure!(self.path, { "a null function pointer" });
                    }
                }
                Ok(true)
            }
            ty::Never => throw_validation_failure!(self.path, { "a value of the never type `!`" }),
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
            | ty::Generator(..) => Ok(false),
            // Some types only occur during typechecking, they have no layout.
            // We should not see them here and we could not check them anyway.
            ty::Error(_)
            | ty::Infer(..)
            | ty::Placeholder(..)
            | ty::Bound(..)
            | ty::Param(..)
            | ty::Opaque(..)
            | ty::Projection(..)
            | ty::GeneratorWitness(..) => bug!("Encountered invalid type {:?}", ty),
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
        let bits = match scalar.try_to_int() {
            Ok(int) => int.assert_bits(size),
            Err(_) => {
                // So this is a pointer then, and casting to an int failed.
                // Can only happen during CTFE.
                // We support 2 kinds of ranges here: full range, and excluding zero.
                if start == 1 && end == max_value {
                    // Only null is the niche.  So make sure the ptr is NOT null.
                    if self.ecx.scalar_may_be_null(scalar)? {
                        throw_validation_failure!(self.path,
                            { "a potentially null pointer" }
                            expected {
                                "something that cannot possibly fail to be {}",
                                wrapping_range_format(valid_range, max_value)
                            }
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
                    throw_validation_failure!(self.path,
                        { "a pointer" }
                        expected {
                            "something that cannot possibly fail to be {}",
                            wrapping_range_format(valid_range, max_value)
                        }
                    )
                }
            }
        };
        // Now compare.
        if valid_range.contains(bits) {
            Ok(())
        } else {
            throw_validation_failure!(self.path,
                { "{}", bits }
                expected { "something {}", wrapping_range_format(valid_range, max_value) }
            )
        }
    }
}

impl<'rt, 'mir, 'tcx: 'mir, M: Machine<'mir, 'tcx>> ValueVisitor<'mir, 'tcx, M>
    for ValidityVisitor<'rt, 'mir, 'tcx, M>
{
    type V = OpTy<'tcx, M::Provenance>;

    #[inline(always)]
    fn ecx(&self) -> &InterpCx<'mir, 'tcx, M> {
        &self.ecx
    }

    fn read_discriminant(
        &mut self,
        op: &OpTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, VariantIdx> {
        self.with_elem(PathElem::EnumTag, move |this| {
            Ok(try_validation!(
                this.ecx.read_discriminant(op),
                this.path,
                InvalidTag(val) =>
                    { "{:x}", val } expected { "a valid enum tag" },
                InvalidUninitBytes(None) =>
                    { "uninitialized bytes" } expected { "a valid enum tag" },
            )
            .1)
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
            // Generators also have variants
            ty::Generator(..) => PathElem::GeneratorState(variant_id),
            _ => bug!("Unexpected type with variant: {:?}", old_op.layout.ty),
        };
        self.with_elem(name, move |this| this.visit_value(new_op))
    }

    #[inline(always)]
    fn visit_union(
        &mut self,
        op: &OpTy<'tcx, M::Provenance>,
        _fields: NonZeroUsize,
    ) -> InterpResult<'tcx> {
        // Special check preventing `UnsafeCell` inside unions in the inner part of constants.
        if matches!(self.ctfe_mode, Some(CtfeValidationMode::Const { inner: true, .. })) {
            if !op.layout.ty.is_freeze(self.ecx.tcx.at(DUMMY_SP), self.ecx.param_env) {
                throw_validation_failure!(self.path, { "`UnsafeCell` in a `const`" });
            }
        }
        Ok(())
    }

    #[inline]
    fn visit_box(&mut self, op: &OpTy<'tcx, M::Provenance>) -> InterpResult<'tcx> {
        self.check_safe_pointer(op, "box")?;
        Ok(())
    }

    #[inline]
    fn visit_value(&mut self, op: &OpTy<'tcx, M::Provenance>) -> InterpResult<'tcx> {
        trace!("visit_value: {:?}, {:?}", *op, op.layout);

        // Check primitive types -- the leaves of our recursive descent.
        if self.try_visit_primitive(op)? {
            return Ok(());
        }

        // Special check preventing `UnsafeCell` in the inner part of constants
        if let Some(def) = op.layout.ty.ty_adt_def() {
            if matches!(self.ctfe_mode, Some(CtfeValidationMode::Const { inner: true, .. }))
                && def.is_unsafe_cell()
            {
                throw_validation_failure!(self.path, { "`UnsafeCell` in a `const`" });
            }
        }

        // Recursively walk the value at its type.
        self.walk_value(op)?;

        // *After* all of this, check the ABI.  We need to check the ABI to handle
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
                throw_validation_failure!(self.path,
                    { "a value of uninhabited type {:?}", op.layout.ty }
                );
            }
            Abi::Scalar(scalar_layout) => {
                if !scalar_layout.is_uninit_valid() {
                    // There is something to check here.
                    let scalar = self.read_scalar(op, "initiailized scalar value")?;
                    self.visit_scalar(scalar, scalar_layout)?;
                }
            }
            Abi::ScalarPair(a_layout, b_layout) => {
                // There is no `rustc_layout_scalar_valid_range_start` for pairs, so
                // we would validate these things as we descend into the fields,
                // but that can miss bugs in layout computation. Layout computation
                // is subtle due to enums having ScalarPair layout, where one field
                // is the discriminant.
                if cfg!(debug_assertions)
                    && !a_layout.is_uninit_valid()
                    && !b_layout.is_uninit_valid()
                {
                    // We can only proceed if *both* scalars need to be initialized.
                    // FIXME: find a way to also check ScalarPair when one side can be uninit but
                    // the other must be init.
                    let (a, b) =
                        self.read_immediate(op, "initiailized scalar value")?.to_scalar_pair();
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

    fn visit_aggregate(
        &mut self,
        op: &OpTy<'tcx, M::Provenance>,
        fields: impl Iterator<Item = InterpResult<'tcx, Self::V>>,
    ) -> InterpResult<'tcx> {
        match op.layout.ty.kind() {
            ty::Str => {
                let mplace = op.assert_mem_place(); // strings are unsized and hence never immediate
                let len = mplace.len(self.ecx)?;
                try_validation!(
                    self.ecx.read_bytes_ptr_strip_provenance(mplace.ptr, Size::from_bytes(len)),
                    self.path,
                    InvalidUninitBytes(..) => { "uninitialized data in `str`" },
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
                let mplace = match op.try_as_mplace() {
                    Ok(mplace) => mplace,
                    Err(imm) => match *imm {
                        Immediate::Uninit =>
                            throw_validation_failure!(self.path, { "uninitialized bytes" }),
                        Immediate::Scalar(..) | Immediate::ScalarPair(..) =>
                            bug!("arrays/slices can never have Scalar/ScalarPair layout"),
                    }
                };

                // Optimization: we just check the entire range at once.
                // NOTE: Keep this in sync with the handling of integer and float
                // types above, in `visit_primitive`.
                // In run-time mode, we accept pointers in here.  This is actually more
                // permissive than a per-element check would be, e.g., we accept
                // a &[u8] that contains a pointer even though bytewise checking would
                // reject it.  However, that's good: We don't inherently want
                // to reject those pointers, we just do not have the machinery to
                // talk about parts of a pointer.
                // We also accept uninit, for consistency with the slow path.
                let alloc = self.ecx.get_ptr_alloc(mplace.ptr, size, mplace.align)?.expect("we already excluded size 0");

                match alloc.get_bytes_strip_provenance() {
                    // In the happy case, we needn't check anything else.
                    Ok(_) => {}
                    // Some error happened, try to provide a more detailed description.
                    Err(err) => {
                        // For some errors we might be able to provide extra information.
                        // (This custom logic does not fit the `try_validation!` macro.)
                        match err.kind() {
                            err_ub!(InvalidUninitBytes(Some((_alloc_id, access)))) => {
                                // Some byte was uninitialized, determine which
                                // element that byte belongs to so we can
                                // provide an index.
                                let i = usize::try_from(
                                    access.uninit.start.bytes() / layout.size.bytes(),
                                )
                                .unwrap();
                                self.path.push(PathElem::ArrayElem(i));

                                throw_validation_failure!(self.path, { "uninitialized bytes" })
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
                self.walk_aggregate(op, fields.take(1))?
            }
            _ => {
                self.walk_aggregate(op, fields)? // default handler
            }
        }
        Ok(())
    }
}

impl<'mir, 'tcx: 'mir, M: Machine<'mir, 'tcx>> InterpCx<'mir, 'tcx, M> {
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
        match visitor.visit_value(&op) {
            Ok(()) => Ok(()),
            // Pass through validation failures.
            Err(err) if matches!(err.kind(), err_ub!(ValidationFailure { .. })) => Err(err),
            // Complain about any other kind of UB error -- those are bad because we'd like to
            // report them in a way that shows *where* in the value the issue lies.
            Err(err) if matches!(err.kind(), InterpError::UndefinedBehavior(_)) => {
                err.print_backtrace();
                bug!("Unexpected Undefined Behavior error during validation: {}", err);
            }
            // Pass through everything else.
            Err(err) => Err(err),
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
    pub fn const_validate_operand(
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
