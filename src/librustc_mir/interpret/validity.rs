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

use syntax_pos::symbol::Symbol;
use rustc::ty::layout::{self, Size, Primitive};
use rustc::ty::{self, Ty};
use rustc_data_structures::fx::FxHashSet;
use rustc::mir::interpret::{
    Scalar, AllocType, EvalResult, ScalarMaybeUndef, EvalErrorKind
};

use super::{
    OpTy, Machine, EvalContext
};

macro_rules! validation_failure{
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

impl<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> EvalContext<'a, 'mir, 'tcx, M> {
    fn validate_scalar(
        &self,
        value: ScalarMaybeUndef,
        size: Size,
        scalar: &layout::Scalar,
        path: &Vec<PathElem>,
        ty: Ty,
    ) -> EvalResult<'tcx> {
        trace!("validate scalar: {:#?}, {:#?}, {:#?}, {}", value, size, scalar, ty);
        let (lo, hi) = scalar.valid_range.clone().into_inner();

        let value = match value {
            ScalarMaybeUndef::Scalar(scalar) => scalar,
            ScalarMaybeUndef::Undef => return validation_failure!("undefined bytes", path),
        };

        let bits = match value {
            Scalar::Bits { bits, size: value_size } => {
                assert_eq!(value_size as u64, size.bytes());
                bits
            },
            Scalar::Ptr(_) => {
                let ptr_size = self.memory.pointer_size();
                let ptr_max = u128::max_value() >> (128 - ptr_size.bits());
                return if lo > hi {
                    if lo - hi == 1 {
                        // no gap, all values are ok
                        Ok(())
                    } else if hi < ptr_max || lo > 1 {
                        let max = u128::max_value() >> (128 - size.bits());
                        validation_failure!(
                            "pointer",
                            path,
                            format!("something in the range {:?} or {:?}", 0..=lo, hi..=max)
                        )
                    } else {
                        Ok(())
                    }
                } else if hi < ptr_max || lo > 1 {
                    validation_failure!(
                        "pointer",
                        path,
                        format!("something in the range {:?}", scalar.valid_range)
                    )
                } else {
                    Ok(())
                };
            },
        };

        // char gets a special treatment, because its number space is not contiguous so `TyLayout`
        // has no special checks for chars
        match ty.sty {
            ty::Char => {
                debug_assert_eq!(size.bytes(), 4);
                if ::std::char::from_u32(bits as u32).is_none() {
                    return validation_failure!(
                        "character",
                        path,
                        "a valid unicode codepoint"
                    );
                }
            }
            _ => {},
        }

        use std::ops::RangeInclusive;
        let in_range = |bound: RangeInclusive<u128>| bound.contains(&bits);
        if lo > hi {
            if in_range(0..=hi) || in_range(lo..=u128::max_value()) {
                Ok(())
            } else {
                validation_failure!(
                    bits,
                    path,
                    format!("something in the range {:?} or {:?}", ..=hi, lo..)
                )
            }
        } else {
            if in_range(scalar.valid_range.clone()) {
                Ok(())
            } else {
                validation_failure!(
                    bits,
                    path,
                    format!("something in the range {:?}", scalar.valid_range)
                )
            }
        }
    }

    /// This function checks the data at `op`.
    /// It will error if the bits at the destination do not match the ones described by the layout.
    /// The `path` may be pushed to, but the part that is present when the function
    /// starts must not be changed!
    pub fn validate_operand(
        &self,
        dest: OpTy<'tcx>,
        path: &mut Vec<PathElem>,
        seen: &mut FxHashSet<(OpTy<'tcx>)>,
        todo: &mut Vec<(OpTy<'tcx>, Vec<PathElem>)>,
    ) -> EvalResult<'tcx> {
        trace!("validate_operand: {:?}, {:#?}", *dest, dest.layout);

        // Find the right variant.  We have to handle this as a prelude, not via
        // proper recursion with the new inner layout, to be able to later nicely
        // print the field names of the enum field that is being accessed.
        let (variant, dest) = match dest.layout.variants {
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
                let inner_dest = self.operand_downcast(dest, variant)?;
                // Put the variant projection onto the path, as a field
                path.push(PathElem::Field(dest.layout.ty
                                          .ty_adt_def()
                                          .unwrap()
                                          .variants[variant].name));
                trace!("variant layout: {:#?}", dest.layout);
                (variant, inner_dest)
            },
            layout::Variants::Single { index } => {
                // Pre-processing for trait objects: Treat them at their real type.
                // (We do not do this for slices and strings: For slices it is not needed,
                // `mplace_array_fields` does the right thing, and for strings there is no
                // real type that would show the actual length.)
                let dest = match dest.layout.ty.sty {
                    ty::Dynamic(..) => {
                        let dest = dest.to_mem_place(); // immediate trait objects are not a thing
                        match self.unpack_dyn_trait(dest) {
                            Ok(res) => res.1.into(),
                            Err(_) =>
                                return validation_failure!(
                                    "invalid vtable in fat pointer", path
                                ),
                        }
                    }
                    _ => dest
                };
                (index, dest)
            }
        };

        // Remember the length, in case we need to truncate
        let path_len = path.len();

        // Validate all fields
        match dest.layout.fields {
            // primitives are unions with zero fields
            // We still check `layout.fields`, not `layout.abi`, because `layout.abi`
            // is `Scalar` for newtypes around scalars, but we want to descend through the
            // fields to get a proper `path`.
            layout::FieldPlacement::Union(0) => {
                match dest.layout.abi {
                    // nothing to do, whatever the pointer points to, it is never going to be read
                    layout::Abi::Uninhabited =>
                        return validation_failure!("a value of an uninhabited type", path),
                    // check that the scalar is a valid pointer or that its bit range matches the
                    // expectation.
                    layout::Abi::Scalar(ref scalar_layout) => {
                        let size = scalar_layout.value.size(self);
                        let value = match self.read_value(dest) {
                            Ok(val) => val,
                            Err(err) => match err.kind {
                                EvalErrorKind::PointerOutOfBounds { .. } |
                                EvalErrorKind::ReadUndefBytes =>
                                    return validation_failure!(
                                        "uninitialized or out-of-bounds memory", path
                                    ),
                                _ =>
                                    return validation_failure!(
                                        "unrepresentable data", path
                                    ),
                            }
                        };
                        let scalar = value.to_scalar_or_undef();
                        self.validate_scalar(scalar, size, scalar_layout, &path, dest.layout.ty)?;
                        if scalar_layout.value == Primitive::Pointer {
                            // ignore integer pointers, we can't reason about the final hardware
                            if let Scalar::Ptr(ptr) = scalar.not_undef()? {
                                let alloc_kind = self.tcx.alloc_map.lock().get(ptr.alloc_id);
                                if let Some(AllocType::Static(did)) = alloc_kind {
                                    // statics from other crates are already checked.
                                    // extern statics cannot be validated as they have no body.
                                    if !did.is_local() || self.tcx.is_foreign_item(did) {
                                        return Ok(());
                                    }
                                }
                                if value.layout.ty.builtin_deref(false).is_some() {
                                    let ptr_op = self.ref_to_mplace(value)?.into();
                                    // we have not encountered this pointer+layout combination
                                    // before.
                                    if seen.insert(ptr_op) {
                                        trace!("Recursing below ptr {:#?}", *value);
                                        todo.push((ptr_op, path_clone_and_deref(path)));
                                    }
                                }
                            }
                        }
                    },
                    _ => bug!("bad abi for FieldPlacement::Union(0): {:#?}", dest.layout.abi),
                }
            }
            layout::FieldPlacement::Union(_) => {
                // We can't check unions, their bits are allowed to be anything.
                // The fields don't need to correspond to any bit pattern of the union's fields.
                // See https://github.com/rust-lang/rust/issues/32836#issuecomment-406875389
            },
            layout::FieldPlacement::Array { .. } if !dest.layout.is_zst() => {
                let dest = dest.to_mem_place(); // non-ZST array/slice/str cannot be immediate
                // Special handling for strings to verify UTF-8
                match dest.layout.ty.sty {
                    ty::Str => {
                        match self.read_str(dest) {
                            Ok(_) => {},
                            Err(err) => match err.kind {
                                EvalErrorKind::PointerOutOfBounds { .. } |
                                EvalErrorKind::ReadUndefBytes =>
                                    // The error here looks slightly different than it does
                                    // for slices, because we do not report the index into the
                                    // str at which we are OOB.
                                    return validation_failure!(
                                        "uninitialized or out-of-bounds memory", path
                                    ),
                                _ =>
                                    return validation_failure!(
                                        "non-UTF-8 data in str", path
                                    ),
                            }
                        }
                    }
                    _ => {
                        // This handles the unsized case correctly as well, as well as
                        // SIMD an all sorts of other array-like types.
                        for (i, field) in self.mplace_array_fields(dest)?.enumerate() {
                            let field = field?;
                            path.push(PathElem::ArrayElem(i));
                            self.validate_operand(field.into(), path, seen, todo)?;
                            path.truncate(path_len);
                        }
                    }
                }
            },
            layout::FieldPlacement::Array { .. } => {
                // An empty array.  Nothing to do.
            }
            layout::FieldPlacement::Arbitrary { ref offsets, .. } => {
                // Fat pointers are treated like pointers, not aggregates.
                if dest.layout.ty.builtin_deref(true).is_some() {
                    // This is a fat pointer.
                    let ptr = match self.read_value(dest.into())
                        .and_then(|val| self.ref_to_mplace(val))
                    {
                        Ok(ptr) => ptr,
                        Err(_) =>
                            return validation_failure!(
                                "undefined location or metadata in fat pointer", path
                            ),
                    };
                    // check metadata early, for better diagnostics
                    match self.tcx.struct_tail(ptr.layout.ty).sty {
                        ty::Dynamic(..) => {
                            match ptr.extra.unwrap().to_ptr() {
                                Ok(_) => {},
                                Err(_) =>
                                    return validation_failure!(
                                        "non-pointer vtable in fat pointer", path
                                    ),
                            }
                        }
                        ty::Slice(..) | ty::Str => {
                            match ptr.extra.unwrap().to_usize(self) {
                                Ok(_) => {},
                                Err(_) =>
                                    return validation_failure!(
                                        "non-integer slice length in fat pointer", path
                                    ),
                            }
                        }
                        _ =>
                            bug!("Unexpected unsized type tail: {:?}",
                                self.tcx.struct_tail(ptr.layout.ty)
                            ),
                    }
                    // for safe ptrs, recursively check it
                    if !dest.layout.ty.is_unsafe_ptr() {
                        let ptr = ptr.into();
                        if seen.insert(ptr) {
                            trace!("Recursing below fat ptr {:?}", ptr);
                            todo.push((ptr, path_clone_and_deref(path)));
                        }
                    }
                } else {
                    // Not a pointer, perform regular aggregate handling below
                    for i in 0..offsets.len() {
                        let field = self.operand_field(dest, i as u64)?;
                        path.push(self.aggregate_field_path_elem(dest.layout.ty, variant, i));
                        self.validate_operand(field, path, seen, todo)?;
                        path.truncate(path_len);
                    }
                }
            }
        }
        Ok(())
    }

    fn aggregate_field_path_elem(&self, ty: Ty<'tcx>, variant: usize, field: usize) -> PathElem {
        match ty.sty {
            // generators and closures.
            ty::Closure(def_id, _) | ty::Generator(def_id, _, _) => {
                let node_id = self.tcx.hir.as_local_node_id(def_id).unwrap();
                let freevar = self.tcx.with_freevars(node_id, |fv| fv[field]);
                PathElem::ClosureVar(self.tcx.hir.name(freevar.var_id()))
            }

            // tuples
            ty::Tuple(_) => PathElem::TupleElem(field),

            // enums
            ty::Adt(def, ..) if def.is_enum() => {
                let variant = &def.variants[variant];
                PathElem::Field(variant.fields[field].ident.name)
            }

            // other ADTs
            ty::Adt(def, _) => PathElem::Field(def.non_enum_variant().fields[field].ident.name),

            // nothing else has an aggregate layout
            _ => bug!("aggregate_field_path_elem: got non-aggregate type {:?}", ty),
        }
    }
}
