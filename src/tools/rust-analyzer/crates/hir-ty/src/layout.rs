//! Compute the binary representation of a type

use base_db::CrateId;
use chalk_ir::{AdtId, TyKind};
use hir_def::{
    layout::{
        Abi, FieldsShape, Integer, Layout, LayoutCalculator, LayoutError, Primitive, ReprOptions,
        RustcEnumVariantIdx, Scalar, Size, StructKind, TargetDataLayout, Variants, WrappingRange,
    },
    LocalFieldId,
};
use stdx::never;

use crate::{db::HirDatabase, Interner, Substitution, Ty};

use self::adt::struct_variant_idx;
pub use self::{
    adt::{layout_of_adt_query, layout_of_adt_recover},
    target::target_data_layout_query,
};

macro_rules! user_error {
    ($x: expr) => {
        return Err(LayoutError::UserError(format!($x)))
    };
}

mod adt;
mod target;

struct LayoutCx<'a> {
    krate: CrateId,
    target: &'a TargetDataLayout,
}

impl<'a> LayoutCalculator for LayoutCx<'a> {
    type TargetDataLayoutRef = &'a TargetDataLayout;

    fn delay_bug(&self, txt: &str) {
        never!("{}", txt);
    }

    fn current_data_layout(&self) -> &'a TargetDataLayout {
        self.target
    }
}

fn scalar_unit(dl: &TargetDataLayout, value: Primitive) -> Scalar {
    Scalar::Initialized { value, valid_range: WrappingRange::full(value.size(dl)) }
}

fn scalar(dl: &TargetDataLayout, value: Primitive) -> Layout {
    Layout::scalar(dl, scalar_unit(dl, value))
}

pub fn layout_of_ty(db: &dyn HirDatabase, ty: &Ty, krate: CrateId) -> Result<Layout, LayoutError> {
    let Some(target) = db.target_data_layout(krate) else { return Err(LayoutError::TargetLayoutNotAvailable) };
    let cx = LayoutCx { krate, target: &target };
    let dl = &*cx.current_data_layout();
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
                    true,
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
                    false,
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

            let fields = tys
                .iter(Interner)
                .map(|k| layout_of_ty(db, k.assert_ty_ref(Interner), krate))
                .collect::<Result<Vec<_>, _>>()?;
            let fields = fields.iter().collect::<Vec<_>>();
            let fields = fields.iter().collect::<Vec<_>>();
            cx.univariant(dl, &fields, &ReprOptions::default(), kind).ok_or(LayoutError::Unknown)?
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
            let element = layout_of_ty(db, element, krate)?;
            let size = element.size.checked_mul(count, dl).ok_or(LayoutError::SizeOverflow)?;

            let abi = if count != 0 && matches!(element.abi, Abi::Uninhabited) {
                Abi::Uninhabited
            } else {
                Abi::Aggregate { sized: true }
            };

            let largest_niche = if count != 0 { element.largest_niche } else { None };

            Layout {
                variants: Variants::Single { index: struct_variant_idx() },
                fields: FieldsShape::Array { stride: element.size, count },
                abi,
                largest_niche,
                align: element.align,
                size,
            }
        }
        TyKind::Slice(element) => {
            let element = layout_of_ty(db, element, krate)?;
            Layout {
                variants: Variants::Single { index: struct_variant_idx() },
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
            cx.scalar_pair(data_ptr, metadata)
        }
        TyKind::FnDef(_, _) => layout_of_unit(&cx, dl)?,
        TyKind::Str => Layout {
            variants: Variants::Single { index: struct_variant_idx() },
            fields: FieldsShape::Array { stride: Size::from_bytes(1), count: 0 },
            abi: Abi::Aggregate { sized: false },
            largest_niche: None,
            align: dl.i8_align,
            size: Size::ZERO,
        },
        TyKind::Never => Layout {
            variants: Variants::Single { index: struct_variant_idx() },
            fields: FieldsShape::Primitive,
            abi: Abi::Uninhabited,
            largest_niche: None,
            align: dl.i8_align,
            size: Size::ZERO,
        },
        TyKind::Dyn(_) | TyKind::Foreign(_) => {
            let mut unit = layout_of_unit(&cx, dl)?;
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
        TyKind::OpaqueType(opaque_ty_id, _) => {
            let impl_trait_id = db.lookup_intern_impl_trait_id((*opaque_ty_id).into());
            match impl_trait_id {
                crate::ImplTraitId::ReturnTypeImplTrait(func, idx) => {
                    let infer = db.infer(func.into());
                    layout_of_ty(db, &infer.type_of_rpit[idx], krate)?
                }
                crate::ImplTraitId::AsyncBlockTypeImplTrait(_, _) => {
                    return Err(LayoutError::NotImplemented)
                }
            }
        }
        TyKind::Closure(_, _) | TyKind::Generator(_, _) | TyKind::GeneratorWitness(_, _) => {
            return Err(LayoutError::NotImplemented)
        }
        TyKind::AssociatedType(_, _)
        | TyKind::Error
        | TyKind::Alias(_)
        | TyKind::Placeholder(_)
        | TyKind::BoundVar(_)
        | TyKind::InferenceVar(_, _) => return Err(LayoutError::HasPlaceholder),
    })
}

fn layout_of_unit(cx: &LayoutCx<'_>, dl: &TargetDataLayout) -> Result<Layout, LayoutError> {
    cx.univariant::<RustcEnumVariantIdx, &&Layout>(
        dl,
        &[],
        &ReprOptions::default(),
        StructKind::AlwaysSized,
    )
    .ok_or(LayoutError::Unknown)
}

fn struct_tail_erasing_lifetimes(db: &dyn HirDatabase, pointee: Ty) -> Ty {
    match pointee.kind(Interner) {
        TyKind::Adt(AdtId(hir_def::AdtId::StructId(i)), subst) => {
            let data = db.struct_data(*i);
            let mut it = data.variant_data.fields().iter().rev();
            match it.next() {
                Some((f, _)) => field_ty(db, (*i).into(), f, subst),
                None => pointee,
            }
        }
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
