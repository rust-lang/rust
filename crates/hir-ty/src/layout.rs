//! Compute the binary representation of a type

use chalk_ir::{AdtId, TyKind};
pub(self) use hir_def::layout::*;
use hir_def::LocalFieldId;

use crate::{db::HirDatabase, Interner, Substitution, Ty};

use self::adt::univariant;
pub use self::{
    adt::{layout_of_adt_query, layout_of_adt_recover},
    target::current_target_data_layout_query,
};

macro_rules! user_error {
    ($x: expr) => {
        return Err(LayoutError::UserError(format!($x)))
    };
}

mod adt;
mod target;

fn scalar_unit(dl: &TargetDataLayout, value: Primitive) -> Scalar {
    Scalar::Initialized { value, valid_range: WrappingRange::full(value.size(dl)) }
}

fn scalar(dl: &TargetDataLayout, value: Primitive) -> Layout {
    Layout::scalar(dl, scalar_unit(dl, value))
}

fn scalar_pair(dl: &TargetDataLayout, a: Scalar, b: Scalar) -> Layout {
    let b_align = b.align(dl);
    let align = a.align(dl).max(b_align).max(dl.aggregate_align);
    let b_offset = a.size(dl).align_to(b_align.abi);
    let size = b_offset.checked_add(b.size(dl), dl).unwrap().align_to(align.abi);

    // HACK(nox): We iter on `b` and then `a` because `max_by_key`
    // returns the last maximum.
    let largest_niche = Niche::from_scalar(dl, b_offset, b)
        .into_iter()
        .chain(Niche::from_scalar(dl, Size::ZERO, a))
        .max_by_key(|niche| niche.available(dl));

    Layout {
        variants: Variants::Single,
        fields: FieldsShape::Arbitrary {
            offsets: vec![Size::ZERO, b_offset],
            memory_index: vec![0, 1],
        },
        abi: Abi::ScalarPair(a, b),
        largest_niche,
        align,
        size,
    }
}

pub fn layout_of_ty(db: &dyn HirDatabase, ty: &Ty) -> Result<Layout, LayoutError> {
    let dl = &*db.current_target_data_layout();
    Ok(match ty.kind(Interner) {
        TyKind::Adt(AdtId(def), subst) => db.layout_of_adt(*def, subst.clone())?,
        TyKind::Scalar(s) => match s {
            chalk_ir::Scalar::Bool => Layout::scalar(
                dl,
                Scalar::Initialized {
                    value: Primitive::Int(Integer::I8, false),
                    valid_range: WrappingRange { start: 0, end: 1 },
                },
            ),
            chalk_ir::Scalar::Char => Layout::scalar(
                dl,
                Scalar::Initialized {
                    value: Primitive::Int(Integer::I32, false),
                    valid_range: WrappingRange { start: 0, end: 0x10FFFF },
                },
            ),
            chalk_ir::Scalar::Int(i) => scalar(
                dl,
                Primitive::Int(
                    match i {
                        chalk_ir::IntTy::Isize => dl.ptr_sized_integer(),
                        chalk_ir::IntTy::I8 => Integer::I8,
                        chalk_ir::IntTy::I16 => Integer::I16,
                        chalk_ir::IntTy::I32 => Integer::I32,
                        chalk_ir::IntTy::I64 => Integer::I64,
                        chalk_ir::IntTy::I128 => Integer::I128,
                    },
                    false,
                ),
            ),
            chalk_ir::Scalar::Uint(i) => scalar(
                dl,
                Primitive::Int(
                    match i {
                        chalk_ir::UintTy::Usize => dl.ptr_sized_integer(),
                        chalk_ir::UintTy::U8 => Integer::I8,
                        chalk_ir::UintTy::U16 => Integer::I16,
                        chalk_ir::UintTy::U32 => Integer::I32,
                        chalk_ir::UintTy::U64 => Integer::I64,
                        chalk_ir::UintTy::U128 => Integer::I128,
                    },
                    true,
                ),
            ),
            chalk_ir::Scalar::Float(f) => scalar(
                dl,
                match f {
                    chalk_ir::FloatTy::F32 => Primitive::F32,
                    chalk_ir::FloatTy::F64 => Primitive::F64,
                },
            ),
        },
        TyKind::Tuple(len, tys) => {
            let kind = if *len == 0 { StructKind::AlwaysSized } else { StructKind::MaybeUnsized };

            univariant(
                dl,
                &tys.iter(Interner)
                    .map(|k| layout_of_ty(db, k.assert_ty_ref(Interner)))
                    .collect::<Result<Vec<_>, _>>()?,
                &ReprOptions::default(),
                kind,
            )?
        }
        TyKind::Array(element, count) => {
            let count = match count.data(Interner).value {
                chalk_ir::ConstValue::Concrete(c) => match c.interned {
                    hir_def::type_ref::ConstScalar::Int(x) => x as u64,
                    hir_def::type_ref::ConstScalar::UInt(x) => x as u64,
                    hir_def::type_ref::ConstScalar::Unknown => {
                        user_error!("unknown const generic parameter")
                    }
                    _ => user_error!("mismatched type of const generic parameter"),
                },
                _ => return Err(LayoutError::HasPlaceholder),
            };
            let element = layout_of_ty(db, element)?;
            let size = element.size.checked_mul(count, dl).ok_or(LayoutError::SizeOverflow)?;

            let abi = if count != 0 && matches!(element.abi, Abi::Uninhabited) {
                Abi::Uninhabited
            } else {
                Abi::Aggregate { sized: true }
            };

            let largest_niche = if count != 0 { element.largest_niche } else { None };

            Layout {
                variants: Variants::Single,
                fields: FieldsShape::Array { stride: element.size, count },
                abi,
                largest_niche,
                align: element.align,
                size,
            }
        }
        TyKind::Slice(element) => {
            let element = layout_of_ty(db, element)?;
            Layout {
                variants: Variants::Single,
                fields: FieldsShape::Array { stride: element.size, count: 0 },
                abi: Abi::Aggregate { sized: false },
                largest_niche: None,
                align: element.align,
                size: Size::ZERO,
            }
        }
        // Potentially-wide pointers.
        TyKind::Ref(_, _, pointee) | TyKind::Raw(_, pointee) => {
            let mut data_ptr = scalar_unit(dl, Primitive::Pointer);
            if matches!(ty.kind(Interner), TyKind::Ref(..)) {
                data_ptr.valid_range_mut().start = 1;
            }

            // let pointee = tcx.normalize_erasing_regions(param_env, pointee);
            // if pointee.is_sized(tcx.at(DUMMY_SP), param_env) {
            //     return Ok(tcx.intern_layout(LayoutS::scalar(cx, data_ptr)));
            // }

            let unsized_part = struct_tail_erasing_lifetimes(db, pointee.clone());
            let metadata = match unsized_part.kind(Interner) {
                TyKind::Slice(_) | TyKind::Str => {
                    scalar_unit(dl, Primitive::Int(dl.ptr_sized_integer(), false))
                }
                TyKind::Dyn(..) => {
                    let mut vtable = scalar_unit(dl, Primitive::Pointer);
                    vtable.valid_range_mut().start = 1;
                    vtable
                }
                _ => {
                    // pointee is sized
                    return Ok(Layout::scalar(dl, data_ptr));
                }
            };

            // Effectively a (ptr, meta) tuple.
            scalar_pair(dl, data_ptr, metadata)
        }
        TyKind::FnDef(_, _) => {
            univariant(dl, &[], &ReprOptions::default(), StructKind::AlwaysSized)?
        }
        TyKind::Str => Layout {
            variants: Variants::Single,
            fields: FieldsShape::Array { stride: Size::from_bytes(1), count: 0 },
            abi: Abi::Aggregate { sized: false },
            largest_niche: None,
            align: dl.i8_align,
            size: Size::ZERO,
        },
        TyKind::Never => Layout {
            variants: Variants::Single,
            fields: FieldsShape::Primitive,
            abi: Abi::Uninhabited,
            largest_niche: None,
            align: dl.i8_align,
            size: Size::ZERO,
        },
        TyKind::Dyn(_) | TyKind::Foreign(_) => {
            let mut unit = univariant(dl, &[], &ReprOptions::default(), StructKind::AlwaysSized)?;
            match unit.abi {
                Abi::Aggregate { ref mut sized } => *sized = false,
                _ => user_error!("bug"),
            }
            unit
        }
        TyKind::Function(_) => {
            let mut ptr = scalar_unit(dl, Primitive::Pointer);
            ptr.valid_range_mut().start = 1;
            Layout::scalar(dl, ptr)
        }
        TyKind::Closure(_, _)
        | TyKind::OpaqueType(_, _)
        | TyKind::Generator(_, _)
        | TyKind::GeneratorWitness(_, _) => return Err(LayoutError::NotImplemented),
        TyKind::AssociatedType(_, _)
        | TyKind::Error
        | TyKind::Alias(_)
        | TyKind::Placeholder(_)
        | TyKind::BoundVar(_)
        | TyKind::InferenceVar(_, _) => return Err(LayoutError::HasPlaceholder),
    })
}

fn struct_tail_erasing_lifetimes(db: &dyn HirDatabase, pointee: Ty) -> Ty {
    match pointee.kind(Interner) {
        TyKind::Adt(AdtId(adt), subst) => match adt {
            &hir_def::AdtId::StructId(i) => {
                let data = db.struct_data(i);
                let mut it = data.variant_data.fields().iter().rev();
                match it.next() {
                    Some((f, _)) => field_ty(db, i.into(), f, subst),
                    None => pointee,
                }
            }
            _ => pointee,
        },
        _ => pointee,
    }
}

fn field_ty(
    db: &dyn HirDatabase,
    def: hir_def::VariantId,
    fd: LocalFieldId,
    subst: &Substitution,
) -> Ty {
    db.field_types(def)[fd].clone().substitute(Interner, subst)
}

#[cfg(test)]
mod tests;
