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
    MPlaceTy, Machine, EvalContext
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

    /// This function checks the memory where `dest` points to.  The place must be sized
    /// (i.e., dest.extra == PlaceExtra::None).
    /// It will error if the bits at the destination do not match the ones described by the layout.
    /// The `path` may be pushed to, but the part that is present when the function
    /// starts must not be changed!
    pub fn validate_mplace(
        &self,
        dest: MPlaceTy<'tcx>,
        path: &mut Vec<PathElem>,
        seen: &mut FxHashSet<(MPlaceTy<'tcx>)>,
        todo: &mut Vec<(MPlaceTy<'tcx>, Vec<PathElem>)>,
    ) -> EvalResult<'tcx> {
        self.memory.dump_alloc(dest.to_ptr()?.alloc_id);
        trace!("validate_mplace: {:?}, {:#?}", *dest, dest.layout);

        // Find the right variant.  We have to handle this as a prelude, not via
        // proper recursion with the new inner layout, to be able to later nicely
        // print the field names of the enum field that is being accessed.
        let (variant, dest) = match dest.layout.variants {
            layout::Variants::NicheFilling { niche: ref tag, .. } |
            layout::Variants::Tagged { ref tag, .. } => {
                let size = tag.value.size(self);
                // we first read the tag value as scalar, to be able to validate it
                let tag_mplace = self.mplace_field(dest, 0)?;
                let tag_value = self.read_scalar(tag_mplace.into())?;
                path.push(PathElem::Tag);
                self.validate_scalar(
                    tag_value, size, tag, &path, tag_mplace.layout.ty
                )?;
                path.pop(); // remove the element again
                // then we read it again to get the index, to continue
                let variant = self.read_discriminant_as_variant_index(dest.into())?;
                let inner_dest = self.mplace_downcast(dest, variant)?;
                // Put the variant projection onto the path, as a field
                path.push(PathElem::Field(dest.layout.ty
                                          .ty_adt_def()
                                          .unwrap()
                                          .variants[variant].name));
                trace!("variant layout: {:#?}", dest.layout);
                (variant, inner_dest)
            },
            layout::Variants::Single { index } => {
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
                        let value = self.read_value(dest.into())?;
                        let scalar = value.to_scalar_or_undef();
                        self.validate_scalar(scalar, size, scalar_layout, &path, dest.layout.ty)?;
                        if scalar_layout.value == Primitive::Pointer {
                            // ignore integer pointers, we can't reason about the final hardware
                            if let Scalar::Ptr(ptr) = scalar.not_undef()? {
                                let alloc_kind = self.tcx.alloc_map.lock().get(ptr.alloc_id);
                                if let Some(AllocType::Static(did)) = alloc_kind {
                                    // statics from other crates are already checked.
                                    // extern statics should not be validated as they have no body.
                                    if !did.is_local() || self.tcx.is_foreign_item(did) {
                                        return Ok(());
                                    }
                                }
                                if value.layout.ty.builtin_deref(false).is_some() {
                                    trace!("Recursing below ptr {:#?}", value);
                                    let ptr_place = self.ref_to_mplace(value)?;
                                    // we have not encountered this pointer+layout
                                    // combination before
                                    if seen.insert(ptr_place) {
                                        todo.push((ptr_place, path_clone_and_deref(path)));
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
            layout::FieldPlacement::Array { .. } => {
                for (i, field) in self.mplace_array_fields(dest)?.enumerate() {
                    let field = field?;
                    path.push(PathElem::ArrayElem(i));
                    self.validate_mplace(field, path, seen, todo)?;
                    path.truncate(path_len);
                }
            },
            layout::FieldPlacement::Arbitrary { ref offsets, .. } => {
                // Fat pointers need special treatment.
                if dest.layout.ty.builtin_deref(true).is_some() {
                    // This is a fat pointer.
                    let ptr = match self.ref_to_mplace(self.read_value(dest.into())?) {
                        Ok(ptr) => ptr,
                        Err(err) => match err.kind {
                            EvalErrorKind::ReadPointerAsBytes =>
                                return validation_failure!(
                                    "fat pointer length is not a valid integer", path
                                ),
                            EvalErrorKind::ReadBytesAsPointer =>
                                return validation_failure!(
                                    "fat pointer vtable is not a valid pointer", path
                                ),
                            _ => return Err(err),
                        }
                    };
                    let unpacked_ptr = self.unpack_unsized_mplace(ptr)?;
                    // for safe ptrs, recursively check it
                    if !dest.layout.ty.is_unsafe_ptr() {
                        trace!("Recursing below fat ptr {:?} (unpacked: {:?})", ptr, unpacked_ptr);
                        if seen.insert(unpacked_ptr) {
                            todo.push((unpacked_ptr, path_clone_and_deref(path)));
                        }
                    }
                } else {
                    // Not a pointer, perform regular aggregate handling below
                    for i in 0..offsets.len() {
                        let field = self.mplace_field(dest, i as u64)?;
                        path.push(self.aggregate_field_path_elem(dest.layout.ty, variant, i));
                        self.validate_mplace(field, path, seen, todo)?;
                        path.truncate(path_len);
                    }
                    // FIXME: For a TyStr, check that this is valid UTF-8.
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
