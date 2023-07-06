//! Compute the binary representation of a type

use base_db::CrateId;
use chalk_ir::{AdtId, FloatTy, IntTy, TyKind, UintTy};
use hir_def::{
    layout::{
        Abi, FieldsShape, Integer, LayoutCalculator, LayoutS, Primitive, ReprOptions, Scalar, Size,
        StructKind, TargetDataLayout, WrappingRange,
    },
    LocalEnumVariantId, LocalFieldId, StructId,
};
use la_arena::{Idx, RawIdx};
use stdx::never;
use triomphe::Arc;

use crate::{
    consteval::try_const_usize, db::HirDatabase, infer::normalize, layout::adt::struct_variant_idx,
    utils::ClosureSubst, Interner, Substitution, TraitEnvironment, Ty,
};

pub use self::{
    adt::{layout_of_adt_query, layout_of_adt_recover},
    target::target_data_layout_query,
};

macro_rules! user_error {
    ($it: expr) => {
        return Err(LayoutError::UserError(format!($it)))
    };
}

mod adt;
mod target;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RustcEnumVariantIdx(pub LocalEnumVariantId);

impl rustc_index::vec::Idx for RustcEnumVariantIdx {
    fn new(idx: usize) -> Self {
        RustcEnumVariantIdx(Idx::from_raw(RawIdx::from(idx as u32)))
    }

    fn index(self) -> usize {
        u32::from(self.0.into_raw()) as usize
    }
}

pub type Layout = LayoutS<RustcEnumVariantIdx>;
pub type TagEncoding = hir_def::layout::TagEncoding<RustcEnumVariantIdx>;
pub type Variants = hir_def::layout::Variants<RustcEnumVariantIdx>;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum LayoutError {
    UserError(String),
    SizeOverflow,
    TargetLayoutNotAvailable,
    HasPlaceholder,
    HasErrorType,
    NotImplemented,
    Unknown,
}

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

// FIXME: move this to the `rustc_abi`.
fn layout_of_simd_ty(
    db: &dyn HirDatabase,
    id: StructId,
    subst: &Substitution,
    krate: CrateId,
    dl: &TargetDataLayout,
) -> Result<Arc<Layout>, LayoutError> {
    let fields = db.field_types(id.into());

    // Supported SIMD vectors are homogeneous ADTs with at least one field:
    //
    // * #[repr(simd)] struct S(T, T, T, T);
    // * #[repr(simd)] struct S { it: T, y: T, z: T, w: T }
    // * #[repr(simd)] struct S([T; 4])
    //
    // where T is a primitive scalar (integer/float/pointer).

    let f0_ty = match fields.iter().next() {
        Some(it) => it.1.clone().substitute(Interner, subst),
        None => {
            user_error!("simd type with zero fields");
        }
    };

    // The element type and number of elements of the SIMD vector
    // are obtained from:
    //
    // * the element type and length of the single array field, if
    // the first field is of array type, or
    //
    // * the homogeneous field type and the number of fields.
    let (e_ty, e_len, is_array) = if let TyKind::Array(e_ty, _) = f0_ty.kind(Interner) {
        // Extract the number of elements from the layout of the array field:
        let FieldsShape::Array { count, .. } = db.layout_of_ty(f0_ty.clone(), krate)?.fields else {
            user_error!("Array with non array layout");
        };

        (e_ty.clone(), count, true)
    } else {
        // First ADT field is not an array:
        (f0_ty, fields.iter().count() as u64, false)
    };

    // Compute the ABI of the element type:
    let e_ly = db.layout_of_ty(e_ty, krate)?;
    let Abi::Scalar(e_abi) = e_ly.abi else {
        user_error!("simd type with inner non scalar type");
    };

    // Compute the size and alignment of the vector:
    let size = e_ly.size.checked_mul(e_len, dl).ok_or(LayoutError::SizeOverflow)?;
    let align = dl.vector_align(size);
    let size = size.align_to(align.abi);

    // Compute the placement of the vector fields:
    let fields = if is_array {
        FieldsShape::Arbitrary { offsets: [Size::ZERO].into(), memory_index: [0].into() }
    } else {
        FieldsShape::Array { stride: e_ly.size, count: e_len }
    };

    Ok(Arc::new(Layout {
        variants: Variants::Single { index: struct_variant_idx() },
        fields,
        abi: Abi::Vector { element: e_abi, count: e_len },
        largest_niche: e_ly.largest_niche,
        size,
        align,
    }))
}

pub fn layout_of_ty_query(
    db: &dyn HirDatabase,
    ty: Ty,
    krate: CrateId,
) -> Result<Arc<Layout>, LayoutError> {
    let Some(target) = db.target_data_layout(krate) else {
        return Err(LayoutError::TargetLayoutNotAvailable);
    };
    let cx = LayoutCx { krate, target: &target };
    let dl = &*cx.current_data_layout();
    let trait_env = Arc::new(TraitEnvironment::empty(krate));
    let ty = normalize(db, trait_env, ty.clone());
    let result = match ty.kind(Interner) {
        TyKind::Adt(AdtId(def), subst) => {
            if let hir_def::AdtId::StructId(s) = def {
                let data = db.struct_data(*s);
                let repr = data.repr.unwrap_or_default();
                if repr.simd() {
                    return layout_of_simd_ty(db, *s, subst, krate, &target);
                }
            };
            return db.layout_of_adt(*def, subst.clone(), krate);
        }
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
                        IntTy::Isize => dl.ptr_sized_integer(),
                        IntTy::I8 => Integer::I8,
                        IntTy::I16 => Integer::I16,
                        IntTy::I32 => Integer::I32,
                        IntTy::I64 => Integer::I64,
                        IntTy::I128 => Integer::I128,
                    },
                    true,
                ),
            ),
            chalk_ir::Scalar::Uint(i) => scalar(
                dl,
                Primitive::Int(
                    match i {
                        UintTy::Usize => dl.ptr_sized_integer(),
                        UintTy::U8 => Integer::I8,
                        UintTy::U16 => Integer::I16,
                        UintTy::U32 => Integer::I32,
                        UintTy::U64 => Integer::I64,
                        UintTy::U128 => Integer::I128,
                    },
                    false,
                ),
            ),
            chalk_ir::Scalar::Float(f) => scalar(
                dl,
                match f {
                    FloatTy::F32 => Primitive::F32,
                    FloatTy::F64 => Primitive::F64,
                },
            ),
        },
        TyKind::Tuple(len, tys) => {
            let kind = if *len == 0 { StructKind::AlwaysSized } else { StructKind::MaybeUnsized };

            let fields = tys
                .iter(Interner)
                .map(|k| db.layout_of_ty(k.assert_ty_ref(Interner).clone(), krate))
                .collect::<Result<Vec<_>, _>>()?;
            let fields = fields.iter().map(|it| &**it).collect::<Vec<_>>();
            let fields = fields.iter().collect::<Vec<_>>();
            cx.univariant(dl, &fields, &ReprOptions::default(), kind).ok_or(LayoutError::Unknown)?
        }
        TyKind::Array(element, count) => {
            let count = try_const_usize(db, &count).ok_or(LayoutError::UserError(
                "unevaluated or mistyped const generic parameter".to_string(),
            ))? as u64;
            let element = db.layout_of_ty(element.clone(), krate)?;
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
            let element = db.layout_of_ty(element.clone(), krate)?;
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
            //     return Ok(tcx.mk_layout(LayoutS::scalar(cx, data_ptr)));
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
                    return Ok(Arc::new(Layout::scalar(dl, data_ptr)));
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
                    return db.layout_of_ty(infer.type_of_rpit[idx].clone(), krate);
                }
                crate::ImplTraitId::AsyncBlockTypeImplTrait(_, _) => {
                    return Err(LayoutError::NotImplemented)
                }
            }
        }
        TyKind::Closure(c, subst) => {
            let (def, _) = db.lookup_intern_closure((*c).into());
            let infer = db.infer(def);
            let (captures, _) = infer.closure_info(c);
            let fields = captures
                .iter()
                .map(|it| {
                    db.layout_of_ty(
                        it.ty.clone().substitute(Interner, ClosureSubst(subst).parent_subst()),
                        krate,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let fields = fields.iter().map(|it| &**it).collect::<Vec<_>>();
            let fields = fields.iter().collect::<Vec<_>>();
            cx.univariant(dl, &fields, &ReprOptions::default(), StructKind::AlwaysSized)
                .ok_or(LayoutError::Unknown)?
        }
        TyKind::Generator(_, _) | TyKind::GeneratorWitness(_, _) => {
            return Err(LayoutError::NotImplemented)
        }
        TyKind::Error => return Err(LayoutError::HasErrorType),
        TyKind::AssociatedType(_, _)
        | TyKind::Alias(_)
        | TyKind::Placeholder(_)
        | TyKind::BoundVar(_)
        | TyKind::InferenceVar(_, _) => return Err(LayoutError::HasPlaceholder),
    };
    Ok(Arc::new(result))
}

pub fn layout_of_ty_recover(
    _: &dyn HirDatabase,
    _: &[String],
    _: &Ty,
    _: &CrateId,
) -> Result<Arc<Layout>, LayoutError> {
    user_error!("infinite sized recursive type");
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
                Some((f, _)) => {
                    let last_field_ty = field_ty(db, (*i).into(), f, subst);
                    struct_tail_erasing_lifetimes(db, last_field_ty)
                }
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

fn scalar_unit(dl: &TargetDataLayout, value: Primitive) -> Scalar {
    Scalar::Initialized { value, valid_range: WrappingRange::full(value.size(dl)) }
}

fn scalar(dl: &TargetDataLayout, value: Primitive) -> Layout {
    Layout::scalar(dl, scalar_unit(dl, value))
}

#[cfg(test)]
mod tests;
