use std::fmt::Write;
use std::ops::RangeInclusive;

use syntax_pos::symbol::{sym, Symbol};
use rustc::hir;
use rustc::ty::layout::{self, TyLayout, LayoutOf, VariantIdx};
use rustc::ty;
use rustc_data_structures::fx::FxHashSet;
use rustc::mir::interpret::{
    GlobalAlloc, InterpResult, InterpError,
};

use std::hash::Hash;

use super::{
    OpTy, Machine, InterpCx, ValueVisitor, MPlaceTy,
};

macro_rules! validation_failure {
    ($what:expr, $where:expr, $details:expr) => {{
        let where_ = path_format(&$where);
        let where_ = if where_.is_empty() {
            String::new()
        } else {
            format!(" at {}", where_)
        };
        err!(ValidationFailure(format!(
            "encountered {}{}, but expected {}",
            $what, where_, $details,
        )))
    }};
    ($what:expr, $where:expr) => {{
        let where_ = path_format(&$where);
        let where_ = if where_.is_empty() {
            String::new()
        } else {
            format!(" at {}", where_)
        };
        err!(ValidationFailure(format!(
            "encountered {}{}",
            $what, where_,
        )))
    }};
}

macro_rules! try_validation {
    ($e:expr, $what:expr, $where:expr, $details:expr) => {{
        match $e {
            Ok(x) => x,
            Err(_) => return validation_failure!($what, $where, $details),
        }
    }};

    ($e:expr, $what:expr, $where:expr) => {{
        match $e {
            Ok(x) => x,
            Err(_) => return validation_failure!($what, $where),
        }
    }}
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
    ClosureVar(Symbol),
    ArrayElem(usize),
    TupleElem(usize),
    Deref,
    Tag,
    DynDowncast,
}

/// State for tracking recursive validation of references
pub struct RefTracking<T, PATH = ()> {
    pub seen: FxHashSet<T>,
    pub todo: Vec<(T, PATH)>,
}

impl<T: Copy + Eq + Hash + std::fmt::Debug, PATH: Default> RefTracking<T, PATH> {
    pub fn empty() -> Self {
        RefTracking {
            seen: FxHashSet::default(),
            todo: vec![],
        }
    }
    pub fn new(op: T) -> Self {
        let mut ref_tracking_for_consts = RefTracking {
            seen: FxHashSet::default(),
            todo: vec![(op, PATH::default())],
        };
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
fn path_format(path: &Vec<PathElem>) -> String {
    use self::PathElem::*;

    let mut out = String::new();
    for elem in path.iter() {
        match elem {
            Field(name) => write!(out, ".{}", name),
            Variant(name) => write!(out, ".<downcast-variant({})>", name),
            GeneratorState(idx) => write!(out, ".<generator-state({})>", idx.index()),
            ClosureVar(name) => write!(out, ".<closure-var({})>", name),
            TupleElem(idx) => write!(out, ".{}", idx),
            ArrayElem(idx) => write!(out, "[{}]", idx),
            Deref =>
                // This does not match Rust syntax, but it is more readable for long paths -- and
                // some of the other items here also are not Rust syntax.  Actually we can't
                // even use the usual syntax because we are just showing the projections,
                // not the root.
                write!(out, ".<deref>"),
            Tag => write!(out, ".<enum-tag>"),
            DynDowncast => write!(out, ".<dyn-downcast>"),
        }.unwrap()
    }
    out
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
    debug_assert!(hi <= max_hi);
    if lo > hi {
        format!("less or equal to {}, or greater or equal to {}", hi, lo)
    } else {
        if lo == 0 {
            debug_assert!(hi < max_hi, "should not be printing if the range covers everything");
            format!("less or equal to {}", hi)
        } else if hi == max_hi {
            format!("greater or equal to {}", lo)
        } else {
            format!("in the range {:?}", r)
        }
    }
}

struct ValidityVisitor<'rt, 'mir, 'tcx, M: Machine<'mir, 'tcx>> {
    /// The `path` may be pushed to, but the part that is present when a function
    /// starts must not be changed!  `visit_fields` and `visit_array` rely on
    /// this stack discipline.
    path: Vec<PathElem>,
    ref_tracking_for_consts: Option<&'rt mut RefTracking<
        MPlaceTy<'tcx, M::PointerTag>,
        Vec<PathElem>,
    >>,
    ecx: &'rt InterpCx<'mir, 'tcx, M>,
}

impl<'rt, 'mir, 'tcx, M: Machine<'mir, 'tcx>> ValidityVisitor<'rt, 'mir, 'tcx, M> {
    fn aggregate_field_path_elem(
        &mut self,
        layout: TyLayout<'tcx>,
        field: usize,
    ) -> PathElem {
        match layout.ty.sty {
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
                                if let hir::PatKind::Binding(_, _, ident, _) = pat.node {
                                    name = Some(ident.name);
                                }
                            }
                        }
                    }
                }

                PathElem::ClosureVar(name.unwrap_or_else(|| {
                    // Fall back to showing the field index.
                    sym::integer(field)
                }))
            }

            // tuples
            ty::Tuple(_) => PathElem::TupleElem(field),

            // enums
            ty::Adt(def, ..) if def.is_enum() => {
                // we might be projecting *to* a variant, or to a field *in*a variant.
                match layout.variants {
                    layout::Variants::Single { index } =>
                        // Inside a variant
                        PathElem::Field(def.variants[index].fields[field].ident.name),
                    _ => bug!(),
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
        new_op: OpTy<'tcx, M::PointerTag>
    ) -> InterpResult<'tcx> {
        let elem = self.aggregate_field_path_elem(old_op.layout, field);
        self.visit_elem(new_op, elem)
    }

    #[inline]
    fn visit_variant(
        &mut self,
        old_op: OpTy<'tcx, M::PointerTag>,
        variant_id: VariantIdx,
        new_op: OpTy<'tcx, M::PointerTag>
    ) -> InterpResult<'tcx> {
        let name = match old_op.layout.ty.sty {
            ty::Adt(adt, _) => PathElem::Variant(adt.variants[variant_id].ident.name),
            // Generators also have variants
            ty::Generator(..) => PathElem::GeneratorState(variant_id),
            _ => bug!("Unexpected type with variant: {:?}", old_op.layout.ty),
        };
        self.visit_elem(new_op, name)
    }

    #[inline]
    fn visit_value(&mut self, op: OpTy<'tcx, M::PointerTag>) -> InterpResult<'tcx>
    {
        trace!("visit_value: {:?}, {:?}", *op, op.layout);
        // Translate some possible errors to something nicer.
        match self.walk_value(op) {
            Ok(()) => Ok(()),
            Err(err) => match err.kind {
                InterpError::InvalidDiscriminant(val) =>
                    validation_failure!(
                        val, self.path, "a valid enum discriminant"
                    ),
                InterpError::ReadPointerAsBytes =>
                    validation_failure!(
                        "a pointer", self.path, "plain (non-pointer) bytes"
                    ),
                _ => Err(err),
            }
        }
    }

    fn visit_primitive(&mut self, value: OpTy<'tcx, M::PointerTag>) -> InterpResult<'tcx>
    {
        let value = self.ecx.read_immediate(value)?;
        // Go over all the primitive types
        let ty = value.layout.ty;
        match ty.sty {
            ty::Bool => {
                let value = value.to_scalar_or_undef();
                try_validation!(value.to_bool(),
                    value, self.path, "a boolean");
            },
            ty::Char => {
                let value = value.to_scalar_or_undef();
                try_validation!(value.to_char(),
                    value, self.path, "a valid unicode codepoint");
            },
            ty::Float(_) | ty::Int(_) | ty::Uint(_) => {
                // NOTE: Keep this in sync with the array optimization for int/float
                // types below!
                let size = value.layout.size;
                let value = value.to_scalar_or_undef();
                if self.ref_tracking_for_consts.is_some() {
                    // Integers/floats in CTFE: Must be scalar bits, pointers are dangerous
                    try_validation!(value.to_bits(size),
                        value, self.path, "initialized plain (non-pointer) bytes");
                } else {
                    // At run-time, for now, we accept *anything* for these types, including
                    // undef. We should fix that, but let's start low.
                }
            }
            ty::RawPtr(..) => {
                if self.ref_tracking_for_consts.is_some() {
                    // Integers/floats in CTFE: For consistency with integers, we do not
                    // accept undef.
                    let _ptr = try_validation!(value.to_scalar_ptr(),
                        "undefined address in raw pointer", self.path);
                    let _meta = try_validation!(value.to_meta(),
                        "uninitialized data in raw fat pointer metadata", self.path);
                } else {
                    // Remain consistent with `usize`: Accept anything.
                }
            }
            _ if ty.is_box() || ty.is_region_ptr() => {
                // Handle fat pointers.
                // Check metadata early, for better diagnostics
                let ptr = try_validation!(value.to_scalar_ptr(),
                    "undefined address in pointer", self.path);
                let meta = try_validation!(value.to_meta(),
                    "uninitialized data in fat pointer metadata", self.path);
                let layout = self.ecx.layout_of(value.layout.ty.builtin_deref(true).unwrap().ty)?;
                if layout.is_unsized() {
                    let tail = self.ecx.tcx.struct_tail(layout.ty);
                    match tail.sty {
                        ty::Dynamic(..) => {
                            let vtable = meta.unwrap();
                            try_validation!(
                                self.ecx.memory.check_ptr_access(
                                    vtable,
                                    3*self.ecx.tcx.data_layout.pointer_size, // drop, size, align
                                    self.ecx.tcx.data_layout.pointer_align.abi,
                                ),
                                "dangling or unaligned vtable pointer or too small vtable",
                                self.path
                            );
                            try_validation!(self.ecx.read_drop_type_from_vtable(vtable),
                                "invalid drop fn in vtable", self.path);
                            try_validation!(self.ecx.read_size_and_align_from_vtable(vtable),
                                "invalid size or align in vtable", self.path);
                            // FIXME: More checks for the vtable.
                        }
                        ty::Slice(..) | ty::Str => {
                            try_validation!(meta.unwrap().to_usize(self.ecx),
                                "non-integer slice length in fat pointer", self.path);
                        }
                        ty::Foreign(..) => {
                            // Unsized, but not fat.
                        }
                        _ =>
                            bug!("Unexpected unsized type tail: {:?}", tail),
                    }
                }
                // Make sure this is dereferencable and all.
                let (size, align) = self.ecx.size_and_align_of(meta, layout)?
                    // for the purpose of validity, consider foreign types to have
                    // alignment and size determined by the layout (size will be 0,
                    // alignment should take attributes into account).
                    .unwrap_or_else(|| (layout.size, layout.align.abi));
                let ptr: Option<_> = match self.ecx.memory.check_ptr_access(ptr, size, align) {
                    Ok(ptr) => ptr,
                    Err(err) => {
                        info!(
                            "{:?} did not pass access check for size {:?}, align {:?}",
                            ptr, size, align
                        );
                        match err.kind {
                            InterpError::InvalidNullPointerUsage =>
                                return validation_failure!("NULL reference", self.path),
                            InterpError::AlignmentCheckFailed { required, has } =>
                                return validation_failure!(format!("unaligned reference \
                                    (required {} byte alignment but found {})",
                                    required.bytes(), has.bytes()), self.path),
                            InterpError::ReadBytesAsPointer =>
                                return validation_failure!(
                                    "integer pointer in non-ZST reference",
                                    self.path
                                ),
                            _ =>
                                return validation_failure!(
                                    "dangling (not entirely in bounds) reference",
                                    self.path
                                ),
                        }
                    }
                };
                // Recursive checking
                if let Some(ref mut ref_tracking) = self.ref_tracking_for_consts {
                    let place = self.ecx.ref_to_mplace(value)?;
                    if let Some(ptr) = ptr { // not a ZST
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
                        }
                    }
                    // Check if we have encountered this pointer+layout combination
                    // before.  Proceed recursively even for ZST, no
                    // reason to skip them! E.g., `!` is a ZST and we want to validate it.
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
            }
            ty::FnPtr(_sig) => {
                let value = value.to_scalar_or_undef();
                let ptr = try_validation!(value.to_ptr(),
                    value, self.path, "a pointer");
                let _fn = try_validation!(self.ecx.memory.get_fn(ptr),
                    value, self.path, "a function pointer");
                // FIXME: Check if the signature matches
            }
            // This should be all the primitive types
            _ => bug!("Unexpected primitive type {}", value.layout.ty)
        }
        Ok(())
    }

    fn visit_uninhabited(&mut self) -> InterpResult<'tcx>
    {
        validation_failure!("a value of an uninhabited type", self.path)
    }

    fn visit_scalar(
        &mut self,
        op: OpTy<'tcx, M::PointerTag>,
        layout: &layout::Scalar,
    ) -> InterpResult<'tcx> {
        let value = self.ecx.read_scalar(op)?;
        // Determine the allowed range
        let (lo, hi) = layout.valid_range.clone().into_inner();
        // `max_hi` is as big as the size fits
        let max_hi = u128::max_value() >> (128 - op.layout.size.bits());
        assert!(hi <= max_hi);
        // We could also write `(hi + 1) % (max_hi + 1) == lo` but `max_hi + 1` overflows for `u128`
        if (lo == 0 && hi == max_hi) || (hi + 1 == lo) {
            // Nothing to check
            return Ok(());
        }
        // At least one value is excluded. Get the bits.
        let value = try_validation!(value.not_undef(),
            value,
            self.path,
            format!(
                "something {}",
                wrapping_range_format(&layout.valid_range, max_hi),
            )
        );
        let bits = match value.to_bits_or_ptr(op.layout.size, self.ecx) {
            Err(ptr) => {
                if lo == 1 && hi == max_hi {
                    // Only NULL is the niche.  So make sure the ptr is NOT NULL.
                    if self.ecx.memory.ptr_may_be_null(ptr) {
                        // These conditions are just here to improve the diagnostics so we can
                        // differentiate between null pointers and dangling pointers
                        if self.ref_tracking_for_consts.is_some() &&
                            self.ecx.memory.get(ptr.alloc_id).is_err() &&
                            self.ecx.memory.get_fn(ptr).is_err() {
                            return validation_failure!(
                                "encountered dangling pointer", self.path
                            );
                        }
                        return validation_failure!("a potentially NULL pointer", self.path);
                    }
                    return Ok(());
                } else {
                    // Conservatively, we reject, because the pointer *could* have this
                    // value.
                    return validation_failure!(
                        "a pointer",
                        self.path,
                        format!(
                            "something that cannot possibly fail to be {}",
                            wrapping_range_format(&layout.valid_range, max_hi)
                        )
                    );
                }
            }
            Ok(data) =>
                data
        };
        // Now compare. This is slightly subtle because this is a special "wrap-around" range.
        if wrapping_range_contains(&layout.valid_range, bits) {
            Ok(())
        } else {
            validation_failure!(
                bits,
                self.path,
                format!("something {}", wrapping_range_format(&layout.valid_range, max_hi))
            )
        }
    }

    fn visit_aggregate(
        &mut self,
        op: OpTy<'tcx, M::PointerTag>,
        fields: impl Iterator<Item=InterpResult<'tcx, Self::V>>,
    ) -> InterpResult<'tcx> {
        match op.layout.ty.sty {
            ty::Str => {
                let mplace = op.to_mem_place(); // strings are never immediate
                try_validation!(self.ecx.read_str(mplace),
                    "uninitialized or non-UTF-8 data in str", self.path);
            }
            ty::Array(tys, ..) | ty::Slice(tys) if {
                // This optimization applies only for integer and floating point types
                // (i.e., types that can hold arbitrary bytes).
                match tys.sty {
                    ty::Int(..) | ty::Uint(..) | ty::Float(..) => true,
                    _ => false,
                }
            } => {
                // bailing out for zsts is ok, since the array element type can only be int/float
                if op.layout.is_zst() {
                    return Ok(());
                }
                // non-ZST array cannot be immediate, slices are never immediate
                let mplace = op.to_mem_place();
                // This is the length of the array/slice.
                let len = mplace.len(self.ecx)?;
                // zero length slices have nothing to be checked
                if len == 0 {
                    return Ok(());
                }
                // This is the element type size.
                let ty_size = self.ecx.layout_of(tys)?.size;
                // This is the size in bytes of the whole array.
                let size = ty_size * len;

                let ptr = self.ecx.force_ptr(mplace.ptr)?;

                // NOTE: Keep this in sync with the handling of integer and float
                // types above, in `visit_primitive`.
                // In run-time mode, we accept pointers in here.  This is actually more
                // permissive than a per-element check would be, e.g., we accept
                // an &[u8] that contains a pointer even though bytewise checking would
                // reject it.  However, that's good: We don't inherently want
                // to reject those pointers, we just do not have the machinery to
                // talk about parts of a pointer.
                // We also accept undef, for consistency with the type-based checks.
                match self.ecx.memory.get(ptr.alloc_id)?.check_bytes(
                    self.ecx,
                    ptr,
                    size,
                    /*allow_ptr_and_undef*/ self.ref_tracking_for_consts.is_none(),
                ) {
                    // In the happy case, we needn't check anything else.
                    Ok(()) => {},
                    // Some error happened, try to provide a more detailed description.
                    Err(err) => {
                        // For some errors we might be able to provide extra information
                        match err.kind {
                            InterpError::ReadUndefBytes(offset) => {
                                // Some byte was undefined, determine which
                                // element that byte belongs to so we can
                                // provide an index.
                                let i = (offset.bytes() / ty_size.bytes()) as usize;
                                self.path.push(PathElem::ArrayElem(i));

                                return validation_failure!(
                                    "undefined bytes", self.path
                                )
                            },
                            // Other errors shouldn't be possible
                            _ => return Err(err),
                        }
                    }
                }
            }
            _ => {
                self.walk_aggregate(op, fields)? // default handler
            }
        }
        Ok(())
    }
}

impl<'mir, 'tcx, M: Machine<'mir, 'tcx>> InterpCx<'mir, 'tcx, M> {
    /// This function checks the data at `op`. `op` is assumed to cover valid memory if it
    /// is an indirect operand.
    /// It will error if the bits at the destination do not match the ones described by the layout.
    ///
    /// `ref_tracking_for_consts` can be `None` to avoid recursive checking below references.
    /// This also toggles between "run-time" (no recursion) and "compile-time" (with recursion)
    /// validation (e.g., pointer values are fine in integers at runtime) and various other const
    /// specific validation checks
    pub fn validate_operand(
        &self,
        op: OpTy<'tcx, M::PointerTag>,
        path: Vec<PathElem>,
        ref_tracking_for_consts: Option<&mut RefTracking<
            MPlaceTy<'tcx, M::PointerTag>,
            Vec<PathElem>,
        >>,
    ) -> InterpResult<'tcx> {
        trace!("validate_operand: {:?}, {:?}", *op, op.layout.ty);

        // Construct a visitor
        let mut visitor = ValidityVisitor {
            path,
            ref_tracking_for_consts,
            ecx: self,
        };

        // Run it
        visitor.visit_value(op)
    }
}
