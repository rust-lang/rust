//! Check the validity invariant of a given value, and tell the user
//! where in the value it got violated.
//! In const context, this goes even further and tries to approximate const safety.
//! That's useful because it means other passes (e.g. promotion) can rely on `const`s
//! to be const-safe.

use std::fmt::Write;
use std::ops::RangeInclusive;

use rustc::ty;
use rustc::ty::layout::{self, LayoutOf, TyLayout, VariantIdx};
use rustc_data_structures::fx::FxHashSet;
use rustc_hir as hir;
use rustc_span::symbol::{sym, Symbol};

use std::hash::Hash;

use super::{
    CheckInAllocMsg, GlobalAlloc, InterpCx, InterpResult, MPlaceTy, Machine, MemPlaceMeta, OpTy,
    ValueVisitor,
};

macro_rules! throw_validation_failure {
    ($what:expr, $where:expr, $details:expr) => {{
        let mut msg = format!("encountered {}", $what);
        let where_ = &$where;
        if !where_.is_empty() {
            msg.push_str(" at ");
            write_path(&mut msg, where_);
        }
        write!(&mut msg, ", but expected {}", $details).unwrap();
        throw_unsup!(ValidationFailure(msg))
    }};
    ($what:expr, $where:expr) => {{
        let mut msg = format!("encountered {}", $what);
        let where_ = &$where;
        if !where_.is_empty() {
            msg.push_str(" at ");
            write_path(&mut msg, where_);
        }
        throw_unsup!(ValidationFailure(msg))
    }};
}

macro_rules! try_validation {
    ($e:expr, $what:expr, $where:expr, $details:expr) => {{
        match $e {
            Ok(x) => x,
            // We re-throw the error, so we are okay with allocation:
            // this can only slow down builds that fail anyway.
            Err(_) => throw_validation_failure!($what, $where, $details),
        }
    }};

    ($e:expr, $what:expr, $where:expr) => {{
        match $e {
            Ok(x) => x,
            // We re-throw the error, so we are okay with allocation:
            // this can only slow down builds that fail anyway.
            Err(_) => throw_validation_failure!($what, $where),
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
fn write_path(out: &mut String, path: &Vec<PathElem>) {
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

// Test if a range that wraps at overflow contains `test`
fn wrapping_range_contains(r: &RangeInclusive<u128>, test: u128) -> bool {
    let (lo, hi) = r.clone().into_inner();
    if lo > hi {
        // Wrapped
        (..=hi).contains(&test) || (lo..).contains(&test)
    } else {
        // Normal
        r.contains(&test)
    }
}

// Formats such that a sentence like "expected something {}" to mean
// "expected something <in the given range>" makes sense.
fn wrapping_range_format(r: &RangeInclusive<u128>, max_hi: u128) -> String {
    let (lo, hi) = r.clone().into_inner();
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
    ref_tracking_for_consts:
        Option<&'rt mut RefTracking<MPlaceTy<'tcx, M::PointerTag>, Vec<PathElem>>>,
    may_ref_to_static: bool,
    ecx: &'rt InterpCx<'mir, 'tcx, M>,
}

impl<'rt, 'mir, 'tcx, M: Machine<'mir, 'tcx>> ValidityVisitor<'rt, 'mir, 'tcx, M> {
    fn aggregate_field_path_elem(&mut self, layout: TyLayout<'tcx>, field: usize) -> PathElem {
        // First, check if we are projecting to a variant.
        match layout.variants {
            layout::Variants::Multiple { discr_index, .. } => {
                if discr_index == field {
                    return match layout.ty.kind {
                        ty::Adt(def, ..) if def.is_enum() => PathElem::EnumTag,
                        ty::Generator(..) => PathElem::GeneratorTag,
                        _ => bug!("non-variant type {:?}", layout.ty),
                    };
                }
            }
            layout::Variants::Single { .. } => {}
        }

        // Now we know we are projecting to a field, so figure out which one.
        match layout.ty.kind {
            // generators and closures.
            ty::Closure(def_id, _) | ty::Generator(def_id, _, _) => {
                let mut name = None;
                if def_id.is_local() {
                    let tables = self.ecx.tcx.typeck_tables_of(def_id);
                    if let Some(upvars) = tables.upvar_list.get(&def_id) {
                        // Sometimes the index is beyond the number of upvars (seen
                        // for a generator).
                        if let Some((&var_hir_id, _)) = upvars.get_index(field) {
                            let node = self.ecx.tcx.hir().get(var_hir_id);
                            if let hir::Node::Binding(pat) = node {
                                if let hir::PatKind::Binding(_, _, ident, _) = pat.kind {
                                    name = Some(ident.name);
                                }
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
                    layout::Variants::Single { index } => {
                        // Inside a variant
                        PathElem::Field(def.variants[index].fields[field].ident.name)
                    }
                    layout::Variants::Multiple { .. } => bug!("we handled variants above"),
                }
            }

            // other ADTs
            ty::Adt(def, _) => PathElem::Field(def.non_enum_variant().fields[field].ident.name),

            // arrays/slices
            ty::Array(..) | ty::Slice(..) => PathElem::ArrayElem(field),

            // dyn traits
            ty::Dynamic(..) => PathElem::DynDowncast,

            // nothing else has an aggregate layout
            _ => bug!("aggregate_field_path_elem: got non-aggregate type {:?}", layout.ty),
        }
    }

    fn visit_elem(
        &mut self,
        new_op: OpTy<'tcx, M::PointerTag>,
        elem: PathElem,
    ) -> InterpResult<'tcx> {
        // Remember the old state
        let path_len = self.path.len();
        // Perform operation
        self.path.push(elem);
        self.visit_value(new_op)?;
        // Undo changes
        self.path.truncate(path_len);
        Ok(())
    }

    fn check_wide_ptr_meta(
        &mut self,
        meta: MemPlaceMeta<M::PointerTag>,
        pointee: TyLayout<'tcx>,
    ) -> InterpResult<'tcx> {
        let tail = self.ecx.tcx.struct_tail_erasing_lifetimes(pointee.ty, self.ecx.param_env);
        match tail.kind {
            ty::Dynamic(..) => {
                let vtable = meta.unwrap_meta();
                try_validation!(
                    self.ecx.memory.check_ptr_access(
                        vtable,
                        3 * self.ecx.tcx.data_layout.pointer_size, // drop, size, align
                        self.ecx.tcx.data_layout.pointer_align.abi,
                    ),
                    "dangling or unaligned vtable pointer in wide pointer or too small vtable",
                    self.path
                );
                try_validation!(
                    self.ecx.read_drop_type_from_vtable(vtable),
                    "invalid drop fn in vtable",
                    self.path
                );
                try_validation!(
                    self.ecx.read_size_and_align_from_vtable(vtable),
                    "invalid size or align in vtable",
                    self.path
                );
                // FIXME: More checks for the vtable.
            }
            ty::Slice(..) | ty::Str => {
                let _len = try_validation!(
                    meta.unwrap_meta().to_machine_usize(self.ecx),
                    "non-integer slice length in wide pointer",
                    self.path
                );
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
        value: OpTy<'tcx, M::PointerTag>,
        kind: &str,
    ) -> InterpResult<'tcx> {
        let value = self.ecx.read_immediate(value)?;
        // Handle wide pointers.
        // Check metadata early, for better diagnostics
        let place = try_validation!(self.ecx.ref_to_mplace(value), "undefined pointer", self.path);
        if place.layout.is_unsized() {
            self.check_wide_ptr_meta(place.meta, place.layout)?;
        }
        // Make sure this is dereferenceable and all.
        let size_and_align = match self.ecx.size_and_align_of(place.meta, place.layout) {
            Ok(res) => res,
            Err(err) => match err.kind {
                err_ub!(InvalidMeta(msg)) => throw_validation_failure!(
                    format_args!("invalid {} metadata: {}", kind, msg),
                    self.path
                ),
                _ => bug!("Unexpected error during ptr size_and_align_of: {}", err),
            },
        };
        let (size, align) = size_and_align
            // for the purpose of validity, consider foreign types to have
            // alignment and size determined by the layout (size will be 0,
            // alignment should take attributes into account).
            .unwrap_or_else(|| (place.layout.size, place.layout.align.abi));
        let ptr: Option<_> = match self.ecx.memory.check_ptr_access_align(
            place.ptr,
            size,
            Some(align),
            CheckInAllocMsg::InboundsTest,
        ) {
            Ok(ptr) => ptr,
            Err(err) => {
                info!(
                    "{:?} did not pass access check for size {:?}, align {:?}",
                    place.ptr, size, align
                );
                match err.kind {
                    err_unsup!(InvalidNullPointerUsage) => {
                        throw_validation_failure!(format_args!("a NULL {}", kind), self.path)
                    }
                    err_unsup!(AlignmentCheckFailed { required, has }) => {
                        throw_validation_failure!(
                            format_args!(
                                "an unaligned {} \
                                    (required {} byte alignment but found {})",
                                kind,
                                required.bytes(),
                                has.bytes()
                            ),
                            self.path
                        )
                    }
                    err_unsup!(ReadBytesAsPointer) => throw_validation_failure!(
                        format_args!("a dangling {} (created from integer)", kind),
                        self.path
                    ),
                    err_unsup!(PointerOutOfBounds { .. }) | err_unsup!(DanglingPointerDeref) => {
                        throw_validation_failure!(
                            format_args!("a dangling {} (not entirely in bounds)", kind),
                            self.path
                        )
                    }
                    _ => bug!("Unexpected error during ptr inbounds test: {}", err),
                }
            }
        };
        // Recursive checking
        if let Some(ref mut ref_tracking) = self.ref_tracking_for_consts {
            if let Some(ptr) = ptr {
                // not a ZST
                // Skip validation entirely for some external statics
                let alloc_kind = self.ecx.tcx.alloc_map.lock().get(ptr.alloc_id);
                if let Some(GlobalAlloc::Static(did)) = alloc_kind {
                    // `extern static` cannot be validated as they have no body.
                    // FIXME: Statics from other crates are also skipped.
                    // They might be checked at a different type, but for now we
                    // want to avoid recursing too deeply.  This is not sound!
                    if !did.is_local() || self.ecx.tcx.is_foreign_item(did) {
                        return Ok(());
                    }
                    if !self.may_ref_to_static && self.ecx.tcx.is_static(did) {
                        throw_validation_failure!(
                            format_args!("a {} pointing to a static variable", kind),
                            self.path
                        );
                    }
                }
            }
            // Proceed recursively even for ZST, no reason to skip them!
            // `!` is a ZST and we want to validate it.
            // Normalize before handing `place` to tracking because that will
            // check for duplicates.
            let place = if size.bytes() > 0 {
                self.ecx.force_mplace_ptr(place).expect("we already bounds-checked")
            } else {
                place
            };
            let path = &self.path;
            ref_tracking.track(place, || {
                // We need to clone the path anyway, make sure it gets created
                // with enough space for the additional `Deref`.
                let mut new_path = Vec::with_capacity(path.len() + 1);
                new_path.clone_from(path);
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
        value: OpTy<'tcx, M::PointerTag>,
    ) -> InterpResult<'tcx, bool> {
        // Go over all the primitive types
        let ty = value.layout.ty;
        match ty.kind {
            ty::Bool => {
                let value = self.ecx.read_scalar(value)?;
                try_validation!(value.to_bool(), value, self.path, "a boolean");
                Ok(true)
            }
            ty::Char => {
                let value = self.ecx.read_scalar(value)?;
                try_validation!(value.to_char(), value, self.path, "a valid unicode codepoint");
                Ok(true)
            }
            ty::Float(_) | ty::Int(_) | ty::Uint(_) => {
                let value = self.ecx.read_scalar(value)?;
                // NOTE: Keep this in sync with the array optimization for int/float
                // types below!
                if self.ref_tracking_for_consts.is_some() {
                    // Integers/floats in CTFE: Must be scalar bits, pointers are dangerous
                    let is_bits = value.not_undef().map_or(false, |v| v.is_bits());
                    if !is_bits {
                        throw_validation_failure!(
                            value,
                            self.path,
                            "initialized plain (non-pointer) bytes"
                        )
                    }
                } else {
                    // At run-time, for now, we accept *anything* for these types, including
                    // undef. We should fix that, but let's start low.
                }
                Ok(true)
            }
            ty::RawPtr(..) => {
                // We are conservative with undef for integers, but try to
                // actually enforce our current rules for raw pointers.
                let place = try_validation!(
                    self.ecx.ref_to_mplace(self.ecx.read_immediate(value)?),
                    "undefined pointer",
                    self.path
                );
                if place.layout.is_unsized() {
                    self.check_wide_ptr_meta(place.meta, place.layout)?;
                }
                Ok(true)
            }
            ty::Ref(..) => {
                self.check_safe_pointer(value, "reference")?;
                Ok(true)
            }
            ty::Adt(def, ..) if def.is_box() => {
                self.check_safe_pointer(value, "box")?;
                Ok(true)
            }
            ty::FnPtr(_sig) => {
                let value = self.ecx.read_scalar(value)?;
                let _fn = try_validation!(
                    value.not_undef().and_then(|ptr| self.ecx.memory.get_fn(ptr)),
                    value,
                    self.path,
                    "a function pointer"
                );
                // FIXME: Check if the signature matches
                Ok(true)
            }
            ty::Never => throw_validation_failure!("a value of the never type `!`", self.path),
            ty::Foreign(..) | ty::FnDef(..) => {
                // Nothing to check.
                Ok(true)
            }
            // The above should be all the (inhabited) primitive types. The rest is compound, we
            // check them by visiting their fields/variants.
            // (`Str` UTF-8 check happens in `visit_aggregate`, too.)
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
            ty::Error
            | ty::Infer(..)
            | ty::Placeholder(..)
            | ty::Bound(..)
            | ty::Param(..)
            | ty::Opaque(..)
            | ty::UnnormalizedProjection(..)
            | ty::Projection(..)
            | ty::GeneratorWitness(..) => bug!("Encountered invalid type {:?}", ty),
        }
    }

    fn visit_scalar(
        &mut self,
        op: OpTy<'tcx, M::PointerTag>,
        scalar_layout: &layout::Scalar,
    ) -> InterpResult<'tcx> {
        let value = self.ecx.read_scalar(op)?;
        let valid_range = &scalar_layout.valid_range;
        let (lo, hi) = valid_range.clone().into_inner();
        // Determine the allowed range
        // `max_hi` is as big as the size fits
        let max_hi = u128::MAX >> (128 - op.layout.size.bits());
        assert!(hi <= max_hi);
        // We could also write `(hi + 1) % (max_hi + 1) == lo` but `max_hi + 1` overflows for `u128`
        if (lo == 0 && hi == max_hi) || (hi + 1 == lo) {
            // Nothing to check
            return Ok(());
        }
        // At least one value is excluded. Get the bits.
        let value = try_validation!(
            value.not_undef(),
            value,
            self.path,
            format_args!("something {}", wrapping_range_format(valid_range, max_hi),)
        );
        let bits = match value.to_bits_or_ptr(op.layout.size, self.ecx) {
            Err(ptr) => {
                if lo == 1 && hi == max_hi {
                    // Only NULL is the niche.  So make sure the ptr is NOT NULL.
                    if self.ecx.memory.ptr_may_be_null(ptr) {
                        throw_validation_failure!(
                            "a potentially NULL pointer",
                            self.path,
                            format_args!(
                                "something that cannot possibly fail to be {}",
                                wrapping_range_format(valid_range, max_hi)
                            )
                        )
                    }
                    return Ok(());
                } else {
                    // Conservatively, we reject, because the pointer *could* have a bad
                    // value.
                    throw_validation_failure!(
                        "a pointer",
                        self.path,
                        format_args!(
                            "something that cannot possibly fail to be {}",
                            wrapping_range_format(valid_range, max_hi)
                        )
                    )
                }
            }
            Ok(data) => data,
        };
        // Now compare. This is slightly subtle because this is a special "wrap-around" range.
        if wrapping_range_contains(&valid_range, bits) {
            Ok(())
        } else {
            throw_validation_failure!(
                bits,
                self.path,
                format_args!("something {}", wrapping_range_format(valid_range, max_hi))
            )
        }
    }
}

impl<'rt, 'mir, 'tcx, M: Machine<'mir, 'tcx>> ValueVisitor<'mir, 'tcx, M>
    for ValidityVisitor<'rt, 'mir, 'tcx, M>
{
    type V = OpTy<'tcx, M::PointerTag>;

    #[inline(always)]
    fn ecx(&self) -> &InterpCx<'mir, 'tcx, M> {
        &self.ecx
    }

    #[inline]
    fn visit_field(
        &mut self,
        old_op: OpTy<'tcx, M::PointerTag>,
        field: usize,
        new_op: OpTy<'tcx, M::PointerTag>,
    ) -> InterpResult<'tcx> {
        let elem = self.aggregate_field_path_elem(old_op.layout, field);
        self.visit_elem(new_op, elem)
    }

    #[inline]
    fn visit_variant(
        &mut self,
        old_op: OpTy<'tcx, M::PointerTag>,
        variant_id: VariantIdx,
        new_op: OpTy<'tcx, M::PointerTag>,
    ) -> InterpResult<'tcx> {
        let name = match old_op.layout.ty.kind {
            ty::Adt(adt, _) => PathElem::Variant(adt.variants[variant_id].ident.name),
            // Generators also have variants
            ty::Generator(..) => PathElem::GeneratorState(variant_id),
            _ => bug!("Unexpected type with variant: {:?}", old_op.layout.ty),
        };
        self.visit_elem(new_op, name)
    }

    #[inline(always)]
    fn visit_union(&mut self, op: OpTy<'tcx, M::PointerTag>, fields: usize) -> InterpResult<'tcx> {
        // Empty unions are not accepted by rustc. But uninhabited enums
        // claim to be unions, so allow them, too.
        assert!(op.layout.abi.is_uninhabited() || fields > 0);
        Ok(())
    }

    #[inline]
    fn visit_value(&mut self, op: OpTy<'tcx, M::PointerTag>) -> InterpResult<'tcx> {
        trace!("visit_value: {:?}, {:?}", *op, op.layout);

        // Check primitive types -- the leafs of our recursive descend.
        if self.try_visit_primitive(op)? {
            return Ok(());
        }
        // Sanity check: `builtin_deref` does not know any pointers that are not primitive.
        assert!(op.layout.ty.builtin_deref(true).is_none());

        // Recursively walk the type. Translate some possible errors to something nicer.
        match self.walk_value(op) {
            Ok(()) => {}
            Err(err) => match err.kind {
                err_ub!(InvalidDiscriminant(val)) => {
                    throw_validation_failure!(val, self.path, "a valid enum discriminant")
                }
                err_unsup!(ReadPointerAsBytes) => {
                    throw_validation_failure!("a pointer", self.path, "plain (non-pointer) bytes")
                }
                // Propagate upwards (that will also check for unexpected errors).
                _ => return Err(err),
            },
        }

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
            layout::Abi::Uninhabited => {
                throw_validation_failure!(
                    format_args!("a value of uninhabited type {:?}", op.layout.ty),
                    self.path
                );
            }
            layout::Abi::Scalar(ref scalar_layout) => {
                self.visit_scalar(op, scalar_layout)?;
            }
            layout::Abi::ScalarPair { .. } | layout::Abi::Vector { .. } => {
                // These have fields that we already visited above, so we already checked
                // all their scalar-level restrictions.
                // There is also no equivalent to `rustc_layout_scalar_valid_range_start`
                // that would make skipping them here an issue.
            }
            layout::Abi::Aggregate { .. } => {
                // Nothing to do.
            }
        }

        Ok(())
    }

    fn visit_aggregate(
        &mut self,
        op: OpTy<'tcx, M::PointerTag>,
        fields: impl Iterator<Item = InterpResult<'tcx, Self::V>>,
    ) -> InterpResult<'tcx> {
        match op.layout.ty.kind {
            ty::Str => {
                let mplace = op.assert_mem_place(self.ecx); // strings are never immediate
                try_validation!(
                    self.ecx.read_str(mplace),
                    "uninitialized or non-UTF-8 data in str",
                    self.path
                );
            }
            ty::Array(tys, ..) | ty::Slice(tys)
                if {
                    // This optimization applies for types that can hold arbitrary bytes (such as
                    // integer and floating point types) or for structs or tuples with no fields.
                    // FIXME(wesleywiser) This logic could be extended further to arbitrary structs
                    // or tuples made up of integer/floating point types or inhabited ZSTs with no
                    // padding.
                    match tys.kind {
                        ty::Int(..) | ty::Uint(..) | ty::Float(..) => true,
                        _ => false,
                    }
                } =>
            {
                // Optimized handling for arrays of integer/float type.

                // Arrays cannot be immediate, slices are never immediate.
                let mplace = op.assert_mem_place(self.ecx);
                // This is the length of the array/slice.
                let len = mplace.len(self.ecx)?;
                // Zero length slices have nothing to be checked.
                if len == 0 {
                    return Ok(());
                }
                // This is the element type size.
                let layout = self.ecx.layout_of(tys)?;
                // This is the size in bytes of the whole array.
                let size = layout.size * len;
                // Size is not 0, get a pointer.
                let ptr = self.ecx.force_ptr(mplace.ptr)?;

                // Optimization: we just check the entire range at once.
                // NOTE: Keep this in sync with the handling of integer and float
                // types above, in `visit_primitive`.
                // In run-time mode, we accept pointers in here.  This is actually more
                // permissive than a per-element check would be, e.g., we accept
                // an &[u8] that contains a pointer even though bytewise checking would
                // reject it.  However, that's good: We don't inherently want
                // to reject those pointers, we just do not have the machinery to
                // talk about parts of a pointer.
                // We also accept undef, for consistency with the slow path.
                match self.ecx.memory.get_raw(ptr.alloc_id)?.check_bytes(
                    self.ecx,
                    ptr,
                    size,
                    /*allow_ptr_and_undef*/ self.ref_tracking_for_consts.is_none(),
                ) {
                    // In the happy case, we needn't check anything else.
                    Ok(()) => {}
                    // Some error happened, try to provide a more detailed description.
                    Err(err) => {
                        // For some errors we might be able to provide extra information
                        match err.kind {
                            err_unsup!(ReadUndefBytes(offset)) => {
                                // Some byte was undefined, determine which
                                // element that byte belongs to so we can
                                // provide an index.
                                let i = (offset.bytes() / layout.size.bytes()) as usize;
                                self.path.push(PathElem::ArrayElem(i));

                                throw_validation_failure!("undefined bytes", self.path)
                            }
                            // Other errors shouldn't be possible
                            _ => return Err(err),
                        }
                    }
                }
            }
            // Fast path for arrays and slices of ZSTs. We only need to check a single ZST element
            // of an array and not all of them, because there's only a single value of a specific
            // ZST type, so either validation fails for all elements or none.
            ty::Array(tys, ..) | ty::Slice(tys) if self.ecx.layout_of(tys)?.is_zst() => {
                // Validate just the first element
                self.walk_aggregate(op, fields.take(1))?
            }
            _ => {
                self.walk_aggregate(op, fields)? // default handler
            }
        }
        Ok(())
    }
}

impl<'mir, 'tcx, M: Machine<'mir, 'tcx>> InterpCx<'mir, 'tcx, M> {
    fn validate_operand_internal(
        &self,
        op: OpTy<'tcx, M::PointerTag>,
        path: Vec<PathElem>,
        ref_tracking_for_consts: Option<
            &mut RefTracking<MPlaceTy<'tcx, M::PointerTag>, Vec<PathElem>>,
        >,
        may_ref_to_static: bool,
    ) -> InterpResult<'tcx> {
        trace!("validate_operand_internal: {:?}, {:?}", *op, op.layout.ty);

        // Construct a visitor
        let mut visitor =
            ValidityVisitor { path, ref_tracking_for_consts, may_ref_to_static, ecx: self };

        // Try to cast to ptr *once* instead of all the time.
        let op = self.force_op_ptr(op).unwrap_or(op);

        // Run it.
        match visitor.visit_value(op) {
            Ok(()) => Ok(()),
            Err(err) if matches!(err.kind, err_unsup!(ValidationFailure { .. })) => Err(err),
            Err(err) if cfg!(debug_assertions) => {
                bug!("Unexpected error during validation: {}", err)
            }
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
    /// `may_ref_to_static` controls whether references are allowed to point to statics.
    #[inline(always)]
    pub fn const_validate_operand(
        &self,
        op: OpTy<'tcx, M::PointerTag>,
        path: Vec<PathElem>,
        ref_tracking: &mut RefTracking<MPlaceTy<'tcx, M::PointerTag>, Vec<PathElem>>,
        may_ref_to_static: bool,
    ) -> InterpResult<'tcx> {
        self.validate_operand_internal(op, path, Some(ref_tracking), may_ref_to_static)
    }

    /// This function checks the data at `op` to be runtime-valid.
    /// `op` is assumed to cover valid memory if it is an indirect operand.
    /// It will error if the bits at the destination do not match the ones described by the layout.
    #[inline(always)]
    pub fn validate_operand(&self, op: OpTy<'tcx, M::PointerTag>) -> InterpResult<'tcx> {
        self.validate_operand_internal(op, vec![], None, false)
    }
}
