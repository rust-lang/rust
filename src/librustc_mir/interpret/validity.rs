// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt::Write;
use std::hash::Hash;

use syntax_pos::symbol::Symbol;
use rustc::ty::layout::{self, Size, Align, TyLayout, LayoutOf};
use rustc::ty;
use rustc_data_structures::fx::FxHashSet;
use rustc::mir::interpret::{
    Scalar, AllocType, EvalResult, EvalErrorKind
};

use super::{
    ValTy, OpTy, MPlaceTy, Machine, EvalContext, ScalarMaybeUndef
};

macro_rules! validation_failure {
    ($what:expr, $where:expr, $details:expr) => {{
        let where_ = path_format($where);
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
        let where_ = path_format($where);
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

/// We want to show a nice path to the invalid field for diagnotsics,
/// but avoid string operations in the happy case where no error happens.
/// So we track a `Vec<PathElem>` where `PathElem` contains all the data we
/// need to later print something for the user.
#[derive(Copy, Clone, Debug)]
pub enum PathElem {
    Field(Symbol),
    ClosureVar(Symbol),
    ArrayElem(usize),
    TupleElem(usize),
    Deref,
    Tag,
}

/// State for tracking recursive validation of references
pub struct RefTracking<'tcx, Tag> {
    pub seen: FxHashSet<(OpTy<'tcx, Tag>)>,
    pub todo: Vec<(OpTy<'tcx, Tag>, Vec<PathElem>)>,
}

impl<'tcx, Tag: Copy+Eq+Hash> RefTracking<'tcx, Tag> {
    pub fn new(op: OpTy<'tcx, Tag>) -> Self {
        let mut ref_tracking = RefTracking {
            seen: FxHashSet::default(),
            todo: vec![(op, Vec::new())],
        };
        ref_tracking.seen.insert(op);
        ref_tracking
    }
}

// Adding a Deref and making a copy of the path to be put into the queue
// always go together.  This one does it with only new allocation.
fn path_clone_and_deref(path: &Vec<PathElem>) -> Vec<PathElem> {
    let mut new_path = Vec::with_capacity(path.len()+1);
    new_path.clone_from(path);
    new_path.push(PathElem::Deref);
    new_path
}

/// Format a path
fn path_format(path: &Vec<PathElem>) -> String {
    use self::PathElem::*;

    let mut out = String::new();
    for elem in path.iter() {
        match elem {
            Field(name) => write!(out, ".{}", name).unwrap(),
            ClosureVar(name) => write!(out, ".<closure-var({})>", name).unwrap(),
            TupleElem(idx) => write!(out, ".{}", idx).unwrap(),
            ArrayElem(idx) => write!(out, "[{}]", idx).unwrap(),
            Deref =>
                // This does not match Rust syntax, but it is more readable for long paths -- and
                // some of the other items here also are not Rust syntax.  Actually we can't
                // even use the usual syntax because we are just showing the projections,
                // not the root.
                write!(out, ".<deref>").unwrap(),
            Tag => write!(out, ".<enum-tag>").unwrap(),
        }
    }
    out
}

fn scalar_format<Tag>(value: ScalarMaybeUndef<Tag>) -> String {
    match value {
        ScalarMaybeUndef::Undef =>
            "uninitialized bytes".to_owned(),
        ScalarMaybeUndef::Scalar(Scalar::Ptr(_)) =>
            "a pointer".to_owned(),
        ScalarMaybeUndef::Scalar(Scalar::Bits { bits, .. }) =>
            bits.to_string(),
    }
}

impl<'a, 'mir, 'tcx, M: Machine<'a, 'mir, 'tcx>> EvalContext<'a, 'mir, 'tcx, M> {
    /// Make sure that `value` is valid for `ty`, *assuming* `ty` is a primitive type.
    fn validate_primitive_type(
        &self,
        value: ValTy<'tcx, M::PointerTag>,
        path: &Vec<PathElem>,
        ref_tracking: Option<&mut RefTracking<'tcx, M::PointerTag>>,
        const_mode: bool,
    ) -> EvalResult<'tcx> {
        // Go over all the primitive types
        let ty = value.layout.ty;
        match ty.sty {
            ty::Bool => {
                let value = value.to_scalar_or_undef();
                try_validation!(value.to_bool(),
                    scalar_format(value), path, "a boolean");
            },
            ty::Char => {
                let value = value.to_scalar_or_undef();
                try_validation!(value.to_char(),
                    scalar_format(value), path, "a valid unicode codepoint");
            },
            ty::Float(_) | ty::Int(_) | ty::Uint(_) => {
                // NOTE: Keep this in sync with the array optimization for int/float
                // types below!
                let size = value.layout.size;
                let value = value.to_scalar_or_undef();
                if const_mode {
                    // Integers/floats in CTFE: Must be scalar bits, pointers are dangerous
                    try_validation!(value.to_bits(size),
                        scalar_format(value), path, "initialized plain bits");
                } else {
                    // At run-time, for now, we accept *anything* for these types, including
                    // undef. We should fix that, but let's start low.
                }
            }
            ty::RawPtr(..) => {
                // No undef allowed here.  Eventually this should be consistent with
                // the integer types.
                let _ptr = try_validation!(value.to_scalar_ptr(),
                    "undefined address in pointer", path);
                let _meta = try_validation!(value.to_meta(),
                    "uninitialized data in fat pointer metadata", path);
            }
            _ if ty.is_box() || ty.is_region_ptr() => {
                // Handle fat pointers.
                // Check metadata early, for better diagnostics
                let ptr = try_validation!(value.to_scalar_ptr(),
                    "undefined address in pointer", path);
                let meta = try_validation!(value.to_meta(),
                    "uninitialized data in fat pointer metadata", path);
                let layout = self.layout_of(value.layout.ty.builtin_deref(true).unwrap().ty)?;
                if layout.is_unsized() {
                    let tail = self.tcx.struct_tail(layout.ty);
                    match tail.sty {
                        ty::Dynamic(..) => {
                            let vtable = try_validation!(meta.unwrap().to_ptr(),
                                "non-pointer vtable in fat pointer", path);
                            try_validation!(self.read_drop_type_from_vtable(vtable),
                                "invalid drop fn in vtable", path);
                            try_validation!(self.read_size_and_align_from_vtable(vtable),
                                "invalid size or align in vtable", path);
                            // FIXME: More checks for the vtable.
                        }
                        ty::Slice(..) | ty::Str => {
                            try_validation!(meta.unwrap().to_usize(self),
                                "non-integer slice length in fat pointer", path);
                        }
                        ty::Foreign(..) => {
                            // Unsized, but not fat.
                        }
                        _ =>
                            bug!("Unexpected unsized type tail: {:?}", tail),
                    }
                }
                // Make sure this is non-NULL and aligned
                let (size, align) = self.size_and_align_of(meta, layout)?
                    // for the purpose of validity, consider foreign types to have
                    // alignment and size determined by the layout (size will be 0,
                    // alignment should take attributes into account).
                    .unwrap_or_else(|| layout.size_and_align());
                match self.memory.check_align(ptr, align) {
                    Ok(_) => {},
                    Err(err) => {
                        error!("{:?} is not aligned to {:?}", ptr, align);
                        match err.kind {
                            EvalErrorKind::InvalidNullPointerUsage =>
                                return validation_failure!("NULL reference", path),
                            EvalErrorKind::AlignmentCheckFailed { .. } =>
                                return validation_failure!("unaligned reference", path),
                            _ =>
                                return validation_failure!(
                                    "dangling (out-of-bounds) reference (might be NULL at \
                                        run-time)",
                                    path
                                ),
                        }
                    }
                }
                // Turn ptr into place.
                // `ref_to_mplace` also calls the machine hook for (re)activating the tag,
                // which in turn will (in full miri) check if the pointer is dereferencable.
                let place = self.ref_to_mplace(value)?;
                // Recursive checking
                if let Some(ref_tracking) = ref_tracking {
                    assert!(const_mode, "We should only do recursie checking in const mode");
                    if size != Size::ZERO {
                        // Non-ZST also have to be dereferencable
                        let ptr = try_validation!(place.ptr.to_ptr(),
                            "integer pointer in non-ZST reference", path);
                        // Skip validation entirely for some external statics
                        let alloc_kind = self.tcx.alloc_map.lock().get(ptr.alloc_id);
                        if let Some(AllocType::Static(did)) = alloc_kind {
                            // `extern static` cannot be validated as they have no body.
                            // FIXME: Statics from other crates are also skipped.
                            // They might be checked at a different type, but for now we
                            // want to avoid recursing too deeply.  This is not sound!
                            if !did.is_local() || self.tcx.is_foreign_item(did) {
                                return Ok(());
                            }
                        }
                        // Maintain the invariant that the place we are checking is
                        // already verified to be in-bounds.
                        try_validation!(self.memory.check_bounds(ptr, size, false),
                            "dangling (not entirely in bounds) reference", path);
                    }
                    // Check if we have encountered this pointer+layout combination
                    // before.  Proceed recursively even for integer pointers, no
                    // reason to skip them! They are (recursively) valid for some ZST,
                    // but not for others (e.g. `!` is a ZST).
                    let op = place.into();
                    if ref_tracking.seen.insert(op) {
                        trace!("Recursing below ptr {:#?}", *op);
                        ref_tracking.todo.push((op, path_clone_and_deref(path)));
                    }
                }
            }
            ty::FnPtr(_sig) => {
                let value = value.to_scalar_or_undef();
                let ptr = try_validation!(value.to_ptr(),
                    scalar_format(value), path, "a pointer");
                let _fn = try_validation!(self.memory.get_fn(ptr),
                    scalar_format(value), path, "a function pointer");
                // FIXME: Check if the signature matches
            }
            // This should be all the primitive types
            ty::Never => bug!("Uninhabited type should have been caught earlier"),
            _ => bug!("Unexpected primitive type {}", value.layout.ty)
        }
        Ok(())
    }

    /// Make sure that `value` matches the
    fn validate_scalar_layout(
        &self,
        value: ScalarMaybeUndef<M::PointerTag>,
        size: Size,
        path: &Vec<PathElem>,
        layout: &layout::Scalar,
    ) -> EvalResult<'tcx> {
        let (lo, hi) = layout.valid_range.clone().into_inner();
        let max_hi = u128::max_value() >> (128 - size.bits()); // as big as the size fits
        assert!(hi <= max_hi);
        if (lo == 0 && hi == max_hi) || (hi + 1 == lo) {
            // Nothing to check
            return Ok(());
        }
        // At least one value is excluded. Get the bits.
        let value = try_validation!(value.not_undef(),
            scalar_format(value), path, format!("something in the range {:?}", layout.valid_range));
        let bits = match value {
            Scalar::Ptr(ptr) => {
                if lo == 1 && hi == max_hi {
                    // only NULL is not allowed.
                    // We can call `check_align` to check non-NULL-ness, but have to also look
                    // for function pointers.
                    let non_null =
                        self.memory.check_align(
                            Scalar::Ptr(ptr), Align::from_bytes(1, 1).unwrap()
                        ).is_ok() ||
                        self.memory.get_fn(ptr).is_ok();
                    if !non_null {
                        // could be NULL
                        return validation_failure!("a potentially NULL pointer", path);
                    }
                    return Ok(());
                } else {
                    // Conservatively, we reject, because the pointer *could* have this
                    // value.
                    return validation_failure!(
                        "a pointer",
                        path,
                        format!(
                            "something that cannot possibly be outside the (wrapping) range {:?}",
                            layout.valid_range
                        )
                    );
                }
            }
            Scalar::Bits { bits, size: value_size } => {
                assert_eq!(value_size as u64, size.bytes());
                bits
            }
        };
        // Now compare. This is slightly subtle because this is a special "wrap-around" range.
        use std::ops::RangeInclusive;
        let in_range = |bound: RangeInclusive<u128>| bound.contains(&bits);
        if lo > hi {
            // wrapping around
            if in_range(0..=hi) || in_range(lo..=max_hi) {
                Ok(())
            } else {
                validation_failure!(
                    bits,
                    path,
                    format!("something in the range {:?} or {:?}", 0..=hi, lo..=max_hi)
                )
            }
        } else {
            if in_range(layout.valid_range.clone()) {
                Ok(())
            } else {
                validation_failure!(
                    bits,
                    path,
                    if hi == max_hi {
                        format!("something greater or equal to {}", lo)
                    } else {
                        format!("something in the range {:?}", layout.valid_range)
                    }
                )
            }
        }
    }

    /// This function checks the data at `op`.  `op` is assumed to cover valid memory if it
    /// is an indirect operand.
    /// It will error if the bits at the destination do not match the ones described by the layout.
    /// The `path` may be pushed to, but the part that is present when the function
    /// starts must not be changed!
    ///
    /// `ref_tracking` can be None to avoid recursive checking below references.
    /// This also toggles between "run-time" (no recursion) and "compile-time" (with recursion)
    /// validation (e.g., pointer values are fine in integers at runtime).
    pub fn validate_operand(
        &self,
        dest: OpTy<'tcx, M::PointerTag>,
        path: &mut Vec<PathElem>,
        mut ref_tracking: Option<&mut RefTracking<'tcx, M::PointerTag>>,
        const_mode: bool,
    ) -> EvalResult<'tcx> {
        trace!("validate_operand: {:?}, {:?}", *dest, dest.layout.ty);

        // If this is a multi-variant layout, we have find the right one and proceed with that.
        // (No good reasoning to make this recursion, but it is equivalent to that.)
        let dest = match dest.layout.variants {
            layout::Variants::NicheFilling { .. } |
            layout::Variants::Tagged { .. } => {
                let variant = match self.read_discriminant(dest) {
                    Ok(res) => res.1,
                    Err(err) => match err.kind {
                        EvalErrorKind::InvalidDiscriminant(val) =>
                            return validation_failure!(
                                format!("invalid enum discriminant {}", val), path
                            ),
                        _ =>
                            return validation_failure!(
                                format!("non-integer enum discriminant"), path
                            ),
                    }
                };
                // Put the variant projection onto the path, as a field
                path.push(PathElem::Field(dest.layout.ty
                                          .ty_adt_def()
                                          .unwrap()
                                          .variants[variant].name));
                // Proceed with this variant
                let dest = self.operand_downcast(dest, variant)?;
                trace!("variant layout: {:#?}", dest.layout);
                dest
            },
            layout::Variants::Single { .. } => dest,
        };

        // First thing, find the real type:
        // If it is a trait object, switch to the actual type that was used to create it.
        let dest = match dest.layout.ty.sty {
            ty::Dynamic(..) => {
                let dest = dest.to_mem_place(); // immediate trait objects are not a thing
                self.unpack_dyn_trait(dest)?.1.into()
            },
            _ => dest
        };

        // If this is a scalar, validate the scalar layout.
        // Things can be aggregates and have scalar layout at the same time, and that
        // is very relevant for `NonNull` and similar structs: We need to validate them
        // at their scalar layout *before* descending into their fields.
        // FIXME: We could avoid some redundant checks here. For newtypes wrapping
        // scalars, we do the same check on every "level" (e.g. first we check
        // MyNewtype and then the scalar in there).
        match dest.layout.abi {
            layout::Abi::Uninhabited =>
                return validation_failure!("a value of an uninhabited type", path),
            layout::Abi::Scalar(ref layout) => {
                let value = try_validation!(self.read_scalar(dest),
                            "uninitialized or unrepresentable data", path);
                self.validate_scalar_layout(value, dest.layout.size, &path, layout)?;
            }
            // FIXME: Should we do something for ScalarPair? Vector?
            _ => {}
        }

        // Check primitive types.  We do this after checking the scalar layout,
        // just to have that done as well.  Primitives can have varying layout,
        // so we check them separately and before aggregate handling.
        // It is CRITICAL that we get this check right, or we might be
        // validating the wrong thing!
        let primitive = match dest.layout.fields {
            // Primitives appear as Union with 0 fields -- except for fat pointers.
            layout::FieldPlacement::Union(0) => true,
            _ => dest.layout.ty.builtin_deref(true).is_some(),
        };
        if primitive {
            let value = try_validation!(self.read_value(dest),
                "uninitialized or unrepresentable data", path);
            return self.validate_primitive_type(
                value,
                &path,
                ref_tracking,
                const_mode,
            );
        }

        // Validate all fields of compound data structures
        let path_len = path.len(); // Remember the length, in case we need to truncate
        match dest.layout.fields {
            layout::FieldPlacement::Union(fields) => {
                // Empty unions are not accepted by rustc. That's great, it means we can
                // use that as an unambiguous signal for detecting primitives.  Make sure
                // we did not miss any primitive.
                debug_assert!(fields > 0);
                // We can't check unions, their bits are allowed to be anything.
                // The fields don't need to correspond to any bit pattern of the union's fields.
                // See https://github.com/rust-lang/rust/issues/32836#issuecomment-406875389
            },
            layout::FieldPlacement::Arbitrary { ref offsets, .. } => {
                // Go look at all the fields
                for i in 0..offsets.len() {
                    let field = self.operand_field(dest, i as u64)?;
                    path.push(self.aggregate_field_path_elem(dest.layout, i));
                    self.validate_operand(
                        field,
                        path,
                        ref_tracking.as_mut().map(|r| &mut **r),
                        const_mode,
                    )?;
                    path.truncate(path_len);
                }
            }
            layout::FieldPlacement::Array { stride, .. } => {
                let dest = if dest.layout.is_zst() {
                    // it's a ZST, the memory content cannot matter
                    MPlaceTy::dangling(dest.layout, self)
                } else {
                    // non-ZST array/slice/str cannot be immediate
                    dest.to_mem_place()
                };
                match dest.layout.ty.sty {
                    // Special handling for strings to verify UTF-8
                    ty::Str => {
                        try_validation!(self.read_str(dest),
                            "uninitialized or non-UTF-8 data in str", path);
                    }
                    // Special handling for arrays/slices of builtin integer types
                    ty::Array(tys, ..) | ty::Slice(tys) if {
                        // This optimization applies only for integer and floating point types
                        // (i.e., types that can hold arbitrary bytes).
                        match tys.sty {
                            ty::Int(..) | ty::Uint(..) | ty::Float(..) => true,
                            _ => false,
                        }
                    } => {
                        // This is the length of the array/slice.
                        let len = dest.len(self)?;
                        // Since primitive types are naturally aligned and tightly packed in arrays,
                        // we can use the stride to get the size of the integral type.
                        let ty_size = stride.bytes();
                        // This is the size in bytes of the whole array.
                        let size = Size::from_bytes(ty_size * len);

                        // NOTE: Keep this in sync with the handling of integer and float
                        // types above, in `validate_primitive_type`.
                        // In run-time mode, we accept pointers in here.  This is actually more
                        // permissive than a per-element check would be, e.g. we accept
                        // an &[u8] that contains a pointer even though bytewise checking would
                        // reject it.  However, that's good: We don't inherently want
                        // to reject those pointers, we just do not have the machinery to
                        // talk about parts of a pointer.
                        // We also accept undef, for consistency with the type-based checks.
                        match self.memory.check_bytes(
                            dest.ptr,
                            size,
                            /*allow_ptr_and_undef*/!const_mode,
                        ) {
                            // In the happy case, we needn't check anything else.
                            Ok(()) => {},
                            // Some error happened, try to provide a more detailed description.
                            Err(err) => {
                                // For some errors we might be able to provide extra information
                                match err.kind {
                                    EvalErrorKind::ReadUndefBytes(offset) => {
                                        // Some byte was undefined, determine which
                                        // element that byte belongs to so we can
                                        // provide an index.
                                        let i = (offset.bytes() / ty_size) as usize;
                                        path.push(PathElem::ArrayElem(i));

                                        return validation_failure!(
                                            "undefined bytes", path
                                        )
                                    },
                                    // Other errors shouldn't be possible
                                    _ => return Err(err),
                                }
                            }
                        }
                    },
                    _ => {
                        // This handles the unsized case correctly as well, as well as
                        // SIMD an all sorts of other array-like types.
                        for (i, field) in self.mplace_array_fields(dest)?.enumerate() {
                            let field = field?;
                            path.push(PathElem::ArrayElem(i));
                            self.validate_operand(
                                field.into(),
                                path,
                                ref_tracking.as_mut().map(|r| &mut **r),
                                const_mode,
                            )?;
                            path.truncate(path_len);
                        }
                    }
                }
            },
        }
        Ok(())
    }

    fn aggregate_field_path_elem(&self, layout: TyLayout<'tcx>, field: usize) -> PathElem {
        match layout.ty.sty {
            // generators and closures.
            ty::Closure(def_id, _) | ty::Generator(def_id, _, _) => {
                if let Some(upvar) = self.tcx.optimized_mir(def_id).upvar_decls.get(field) {
                    PathElem::ClosureVar(upvar.debug_name)
                } else {
                    // Sometimes the index is beyond the number of freevars (seen
                    // for a generator).
                    PathElem::ClosureVar(Symbol::intern(&field.to_string()))
                }
            }

            // tuples
            ty::Tuple(_) => PathElem::TupleElem(field),

            // enums
            ty::Adt(def, ..) if def.is_enum() => {
                let variant = match layout.variants {
                    layout::Variants::Single { index } => &def.variants[index],
                    _ => bug!("aggregate_field_path_elem: got enum but not in a specific variant"),
                };
                PathElem::Field(variant.fields[field].ident.name)
            }

            // other ADTs
            ty::Adt(def, _) => PathElem::Field(def.non_enum_variant().fields[field].ident.name),

            // nothing else has an aggregate layout
            _ => bug!("aggregate_field_path_elem: got non-aggregate type {:?}", layout.ty),
        }
    }
}
