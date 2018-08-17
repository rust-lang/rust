use std::fmt::Write;

use rustc::ty::layout::{self, Size, Primitive};
use rustc::ty::{self, Ty};
use rustc_data_structures::fx::FxHashSet;
use rustc::mir::interpret::{
    Scalar, AllocType, EvalResult, ScalarMaybeUndef,
};

use super::{
    MPlaceTy, PlaceExtra, Machine, EvalContext
};

macro_rules! validation_failure{
    ($what:expr, $where:expr, $details:expr) => {{
        let where_ = if $where.is_empty() {
            String::new()
        } else {
            format!(" at {}", $where)
        };
        err!(ValidationFailure(format!(
            "encountered {}{}, but expected {}",
            $what, where_, $details,
        )))
    }};
    ($what:expr, $where:expr) => {{
        let where_ = if $where.is_empty() {
            String::new()
        } else {
            format!(" at {}", $where)
        };
        err!(ValidationFailure(format!(
            "encountered {}{}",
            $what, where_,
        )))
    }};
}

impl<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> EvalContext<'a, 'mir, 'tcx, M> {
    fn validate_scalar(
        &self,
        value: ScalarMaybeUndef,
        size: Size,
        scalar: &layout::Scalar,
        path: &str,
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
            ty::TyChar => {
                debug_assert_eq!(size.bytes(), 4);
                if ::std::char::from_u32(bits as u32).is_none() {
                    return err!(InvalidChar(bits));
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

    /// This function checks the memory where `ptr` points to.
    /// It will error if the bits at the destination do not match the ones described by the layout.
    pub fn validate_mplace(
        &self,
        dest: MPlaceTy<'tcx>,
        path: String,
        seen: &mut FxHashSet<(MPlaceTy<'tcx>)>,
        todo: &mut Vec<(MPlaceTy<'tcx>, String)>,
    ) -> EvalResult<'tcx> {
        self.memory.dump_alloc(dest.to_ptr()?.alloc_id);
        trace!("validate_mplace: {:?}, {:#?}", *dest, dest.layout);

        // Find the right variant
        let (variant, dest) = match dest.layout.variants {
            layout::Variants::NicheFilling { niche: ref tag, .. } |
            layout::Variants::Tagged { ref tag, .. } => {
                let size = tag.value.size(self);
                // we first read the tag value as scalar, to be able to validate it
                let tag_mplace = self.mplace_field(dest, 0)?;
                let tag_value = self.read_scalar(tag_mplace.into())?;
                let path = format!("{}.TAG", path);
                self.validate_scalar(
                    tag_value, size, tag, &path, tag_mplace.layout.ty
                )?;
                // then we read it again to get the index, to continue
                let variant = self.read_discriminant_as_variant_index(dest.into())?;
                let dest = self.mplace_downcast(dest, variant)?;
                trace!("variant layout: {:#?}", dest.layout);
                (variant, dest)
            },
            layout::Variants::Single { index } => {
                (index, dest)
            }
        };

        // Validate all fields
        match dest.layout.fields {
            // primitives are unions with zero fields
            layout::FieldPlacement::Union(0) => {
                match dest.layout.abi {
                    // nothing to do, whatever the pointer points to, it is never going to be read
                    layout::Abi::Uninhabited => validation_failure!("a value of an uninhabited type", path),
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
                                    // statics from other crates are already checked
                                    // extern statics should not be validated as they have no body
                                    if !did.is_local() || self.tcx.is_foreign_item(did) {
                                        return Ok(());
                                    }
                                }
                                if value.layout.ty.builtin_deref(false).is_some() {
                                    trace!("Recursing below ptr {:#?}", value);
                                    let ptr_place = self.ref_to_mplace(value)?;
                                    // we have not encountered this pointer+layout combination before
                                    if seen.insert(ptr_place) {
                                        todo.push((ptr_place, format!("(*{})", path)))
                                    }
                                }
                            }
                        }
                        Ok(())
                    },
                    _ => bug!("bad abi for FieldPlacement::Union(0): {:#?}", dest.layout.abi),
                }
            }
            layout::FieldPlacement::Union(_) => {
                // We can't check unions, their bits are allowed to be anything.
                // The fields don't need to correspond to any bit pattern of the union's fields.
                // See https://github.com/rust-lang/rust/issues/32836#issuecomment-406875389
                Ok(())
            },
            layout::FieldPlacement::Array { count, .. } => {
                for i in 0..count {
                    let mut path = path.clone();
                    self.dump_field_name(&mut path, dest.layout.ty, i as usize, variant).unwrap();
                    let field = self.mplace_field(dest, i)?;
                    self.validate_mplace(field, path, seen, todo)?;
                }
                Ok(())
            },
            layout::FieldPlacement::Arbitrary { ref offsets, .. } => {
                // fat pointers need special treatment
                match dest.layout.ty.builtin_deref(false).map(|tam| &tam.ty.sty) {
                    | Some(ty::TyStr)
                    | Some(ty::TySlice(_)) => {
                        // check the length (for nicer error messages)
                        let len_mplace = self.mplace_field(dest, 1)?;
                        let len = self.read_scalar(len_mplace.into())?;
                        let len = match len.to_bits(len_mplace.layout.size) {
                            Err(_) => return validation_failure!("length is not a valid integer", path),
                            Ok(len) => len as u64,
                        };
                        // get the fat ptr, and recursively check it
                        let ptr = self.ref_to_mplace(self.read_value(dest.into())?)?;
                        assert_eq!(ptr.extra, PlaceExtra::Length(len));
                        let unpacked_ptr = self.unpack_unsized_mplace(ptr)?;
                        if seen.insert(unpacked_ptr) {
                            let mut path = path.clone();
                            self.dump_field_name(&mut path, dest.layout.ty, 0, 0).unwrap();
                            todo.push((unpacked_ptr, path))
                        }
                    },
                    Some(ty::TyDynamic(..)) => {
                        // check the vtable (for nicer error messages)
                        let vtable = self.read_scalar(self.mplace_field(dest, 1)?.into())?;
                        let vtable = match vtable.to_ptr() {
                            Err(_) => return validation_failure!("vtable address is not a pointer", path),
                            Ok(vtable) => vtable,
                        };
                        // get the fat ptr, and recursively check it
                        let ptr = self.ref_to_mplace(self.read_value(dest.into())?)?;
                        assert_eq!(ptr.extra, PlaceExtra::Vtable(vtable));
                        let unpacked_ptr = self.unpack_unsized_mplace(ptr)?;
                        if seen.insert(unpacked_ptr) {
                            let mut path = path.clone();
                            self.dump_field_name(&mut path, dest.layout.ty, 0, 0).unwrap();
                            todo.push((unpacked_ptr, path))
                        }
                        // FIXME: More checks for the vtable... making sure it is exactly
                        // the one one would expect for this type.
                    },
                    Some(ty) =>
                        bug!("Unexpected fat pointer target type {:?}", ty),
                    None => {
                        // Not a pointer, perform regular aggregate handling below
                        for i in 0..offsets.len() {
                            let mut path = path.clone();
                            self.dump_field_name(&mut path, dest.layout.ty, i, variant).unwrap();
                            let field = self.mplace_field(dest, i as u64)?;
                            self.validate_mplace(field, path, seen, todo)?;
                        }
                        // FIXME: For a TyStr, check that this is valid UTF-8.
                    },
                }

                Ok(())
            }
        }
    }

    fn dump_field_name(&self, s: &mut String, ty: Ty<'tcx>, i: usize, variant: usize) -> ::std::fmt::Result {
        match ty.sty {
            ty::TyBool |
            ty::TyChar |
            ty::TyInt(_) |
            ty::TyUint(_) |
            ty::TyFloat(_) |
            ty::TyFnPtr(_) |
            ty::TyNever |
            ty::TyFnDef(..) |
            ty::TyGeneratorWitness(..) |
            ty::TyForeign(..) |
            ty::TyDynamic(..) => {
                bug!("field_name({:?}): not applicable", ty)
            }

            // Potentially-fat pointers.
            ty::TyRef(_, pointee, _) |
            ty::TyRawPtr(ty::TypeAndMut { ty: pointee, .. }) => {
                assert!(i < 2);

                // Reuse the fat *T type as its own thin pointer data field.
                // This provides information about e.g. DST struct pointees
                // (which may have no non-DST form), and will work as long
                // as the `Abi` or `FieldPlacement` is checked by users.
                if i == 0 {
                    return write!(s, ".data_ptr");
                }

                match self.tcx.struct_tail(pointee).sty {
                    ty::TySlice(_) |
                    ty::TyStr => write!(s, ".len"),
                    ty::TyDynamic(..) => write!(s, ".vtable_ptr"),
                    _ => bug!("field_name({:?}): not applicable", ty)
                }
            }

            // Arrays and slices.
            ty::TyArray(_, _) |
            ty::TySlice(_) |
            ty::TyStr => write!(s, "[{}]", i),

            // generators and closures.
            ty::TyClosure(def_id, _) | ty::TyGenerator(def_id, _, _) => {
                let node_id = self.tcx.hir.as_local_node_id(def_id).unwrap();
                let freevar = self.tcx.with_freevars(node_id, |fv| fv[i]);
                write!(s, ".upvar({})", self.tcx.hir.name(freevar.var_id()))
            }

            ty::TyTuple(_) => write!(s, ".{}", i),

            // enums
            ty::TyAdt(def, ..) if def.is_enum() => {
                let variant = &def.variants[variant];
                write!(s, ".{}::{}", variant.name, variant.fields[i].ident)
            }

            // other ADTs.
            ty::TyAdt(def, _) => write!(s, ".{}", def.non_enum_variant().fields[i].ident),

            ty::TyProjection(_) | ty::TyAnon(..) | ty::TyParam(_) |
            ty::TyInfer(_) | ty::TyError => {
                bug!("dump_field_name: unexpected type `{}`", ty)
            }
        }
    }
}
