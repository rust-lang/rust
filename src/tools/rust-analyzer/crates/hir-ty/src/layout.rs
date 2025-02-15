//! Compute the binary representation of a type

use std::fmt;

use base_db::ra_salsa::Cycle;
use chalk_ir::{AdtId, FloatTy, IntTy, TyKind, UintTy};
use hir_def::{
    layout::{
        BackendRepr, FieldsShape, Float, Integer, LayoutCalculator, LayoutCalculatorError,
        LayoutData, Primitive, ReprOptions, Scalar, Size, StructKind, TargetDataLayout,
        WrappingRange,
    },
    LocalFieldId, StructId,
};
use la_arena::{Idx, RawIdx};
use rustc_abi::AddressSpace;
use rustc_index::{IndexSlice, IndexVec};
use rustc_hashes::Hash64;

use triomphe::Arc;

use crate::{
    consteval::try_const_usize,
    db::{HirDatabase, InternedClosure},
    infer::normalize,
    layout::adt::struct_variant_idx,
    utils::ClosureSubst,
    Interner, ProjectionTy, Substitution, TraitEnvironment, Ty,
};

pub use self::{
    adt::{layout_of_adt_query, layout_of_adt_recover},
    target::target_data_layout_query,
};

mod adt;
mod target;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RustcEnumVariantIdx(pub usize);

impl rustc_index::Idx for RustcEnumVariantIdx {
    fn new(idx: usize) -> Self {
        RustcEnumVariantIdx(idx)
    }

    fn index(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RustcFieldIdx(pub LocalFieldId);

impl RustcFieldIdx {
    pub fn new(idx: usize) -> Self {
        RustcFieldIdx(Idx::from_raw(RawIdx::from(idx as u32)))
    }
}

impl rustc_index::Idx for RustcFieldIdx {
    fn new(idx: usize) -> Self {
        RustcFieldIdx(Idx::from_raw(RawIdx::from(idx as u32)))
    }

    fn index(self) -> usize {
        u32::from(self.0.into_raw()) as usize
    }
}

pub type Layout = LayoutData<RustcFieldIdx, RustcEnumVariantIdx>;
pub type TagEncoding = hir_def::layout::TagEncoding<RustcEnumVariantIdx>;
pub type Variants = hir_def::layout::Variants<RustcFieldIdx, RustcEnumVariantIdx>;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum LayoutError {
    // FIXME: Remove more variants once they get added to LayoutCalculatorError
    BadCalc(LayoutCalculatorError<()>),
    HasErrorConst,
    HasErrorType,
    HasPlaceholder,
    InvalidSimdType,
    NotImplemented,
    RecursiveTypeWithoutIndirection,
    TargetLayoutNotAvailable,
    Unknown,
    UserReprTooSmall,
}

impl std::error::Error for LayoutError {}
impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayoutError::BadCalc(err) => err.fallback_fmt(f),
            LayoutError::HasErrorConst => write!(f, "type contains an unevaluatable const"),
            LayoutError::HasErrorType => write!(f, "type contains an error"),
            LayoutError::HasPlaceholder => write!(f, "type contains placeholders"),
            LayoutError::InvalidSimdType => write!(f, "invalid simd type definition"),
            LayoutError::NotImplemented => write!(f, "not implemented"),
            LayoutError::RecursiveTypeWithoutIndirection => {
                write!(f, "recursive type without indirection")
            }
            LayoutError::TargetLayoutNotAvailable => write!(f, "target layout not available"),
            LayoutError::Unknown => write!(f, "unknown"),
            LayoutError::UserReprTooSmall => {
                write!(f, "the `#[repr]` hint is too small to hold the discriminants of the enum")
            }
        }
    }
}

impl<F> From<LayoutCalculatorError<F>> for LayoutError {
    fn from(err: LayoutCalculatorError<F>) -> Self {
        LayoutError::BadCalc(err.without_payload())
    }
}

struct LayoutCx<'a> {
    calc: LayoutCalculator<&'a TargetDataLayout>,
}

impl<'a> LayoutCx<'a> {
    fn new(target: &'a TargetDataLayout) -> Self {
        Self { calc: LayoutCalculator::new(target) }
    }
}

// FIXME: move this to the `rustc_abi`.
fn layout_of_simd_ty(
    db: &dyn HirDatabase,
    id: StructId,
    subst: &Substitution,
    env: Arc<TraitEnvironment>,
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
        None => return Err(LayoutError::InvalidSimdType),
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
        let FieldsShape::Array { count, .. } = db.layout_of_ty(f0_ty.clone(), env.clone())?.fields
        else {
            return Err(LayoutError::Unknown);
        };

        (e_ty.clone(), count, true)
    } else {
        // First ADT field is not an array:
        (f0_ty, fields.iter().count() as u64, false)
    };

    // Compute the ABI of the element type:
    let e_ly = db.layout_of_ty(e_ty, env)?;
    let BackendRepr::Scalar(e_abi) = e_ly.backend_repr else {
        return Err(LayoutError::Unknown);
    };

    // Compute the size and alignment of the vector:
    let size = e_ly
        .size
        .checked_mul(e_len, dl)
        .ok_or(LayoutError::BadCalc(LayoutCalculatorError::SizeOverflow))?;
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
        backend_repr: BackendRepr::Vector { element: e_abi, count: e_len },
        largest_niche: e_ly.largest_niche,
        size,
        align,
        max_repr_align: None,
        unadjusted_abi_align: align.abi,
        randomization_seed: Hash64::ZERO,
    }))
}

pub fn layout_of_ty_query(
    db: &dyn HirDatabase,
    ty: Ty,
    trait_env: Arc<TraitEnvironment>,
) -> Result<Arc<Layout>, LayoutError> {
    let krate = trait_env.krate;
    let Ok(target) = db.target_data_layout(krate) else {
        return Err(LayoutError::TargetLayoutNotAvailable);
    };
    let dl = &*target;
    let cx = LayoutCx::new(dl);
    let ty = normalize(db, trait_env.clone(), ty);
    let result = match ty.kind(Interner) {
        TyKind::Adt(AdtId(def), subst) => {
            if let hir_def::AdtId::StructId(s) = def {
                let data = db.struct_data(*s);
                let repr = data.repr.unwrap_or_default();
                if repr.simd() {
                    return layout_of_simd_ty(db, *s, subst, trait_env, &target);
                }
            };
            return db.layout_of_adt(*def, subst.clone(), trait_env);
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
                Primitive::Float(match f {
                    FloatTy::F16 => Float::F16,
                    FloatTy::F32 => Float::F32,
                    FloatTy::F64 => Float::F64,
                    FloatTy::F128 => Float::F128,
                }),
            ),
        },
        TyKind::Tuple(len, tys) => {
            let kind = if *len == 0 { StructKind::AlwaysSized } else { StructKind::MaybeUnsized };

            let fields = tys
                .iter(Interner)
                .map(|k| db.layout_of_ty(k.assert_ty_ref(Interner).clone(), trait_env.clone()))
                .collect::<Result<Vec<_>, _>>()?;
            let fields = fields.iter().map(|it| &**it).collect::<Vec<_>>();
            let fields = fields.iter().collect::<IndexVec<_, _>>();
            cx.calc.univariant(&fields, &ReprOptions::default(), kind)?
        }
        TyKind::Array(element, count) => {
            let count = try_const_usize(db, count).ok_or(LayoutError::HasErrorConst)? as u64;
            let element = db.layout_of_ty(element.clone(), trait_env)?;
            let size = element
                .size
                .checked_mul(count, dl)
                .ok_or(LayoutError::BadCalc(LayoutCalculatorError::SizeOverflow))?;

            let backend_repr =
                if count != 0 && matches!(element.backend_repr, BackendRepr::Uninhabited) {
                    BackendRepr::Uninhabited
                } else {
                    BackendRepr::Memory { sized: true }
                };

            let largest_niche = if count != 0 { element.largest_niche } else { None };

            Layout {
                variants: Variants::Single { index: struct_variant_idx() },
                fields: FieldsShape::Array { stride: element.size, count },
                backend_repr,
                largest_niche,
                align: element.align,
                size,
                max_repr_align: None,
                unadjusted_abi_align: element.align.abi,
                randomization_seed: Hash64::ZERO,
            }
        }
        TyKind::Slice(element) => {
            let element = db.layout_of_ty(element.clone(), trait_env)?;
            Layout {
                variants: Variants::Single { index: struct_variant_idx() },
                fields: FieldsShape::Array { stride: element.size, count: 0 },
                backend_repr: BackendRepr::Memory { sized: false },
                largest_niche: None,
                align: element.align,
                size: Size::ZERO,
                max_repr_align: None,
                unadjusted_abi_align: element.align.abi,
                randomization_seed: Hash64::ZERO,
            }
        }
        TyKind::Str => Layout {
            variants: Variants::Single { index: struct_variant_idx() },
            fields: FieldsShape::Array { stride: Size::from_bytes(1), count: 0 },
            backend_repr: BackendRepr::Memory { sized: false },
            largest_niche: None,
            align: dl.i8_align,
            size: Size::ZERO,
            max_repr_align: None,
            unadjusted_abi_align: dl.i8_align.abi,
            randomization_seed: Hash64::ZERO,
        },
        // Potentially-wide pointers.
        TyKind::Ref(_, _, pointee) | TyKind::Raw(_, pointee) => {
            let mut data_ptr = scalar_unit(dl, Primitive::Pointer(AddressSpace::DATA));
            if matches!(ty.kind(Interner), TyKind::Ref(..)) {
                data_ptr.valid_range_mut().start = 1;
            }

            // let pointee = tcx.normalize_erasing_regions(param_env, pointee);
            // if pointee.is_sized(tcx.at(DUMMY_SP), param_env) {
            //     return Ok(tcx.mk_layout(LayoutS::scalar(cx, data_ptr)));
            // }

            let mut unsized_part = struct_tail_erasing_lifetimes(db, pointee.clone());
            if let TyKind::AssociatedType(id, subst) = unsized_part.kind(Interner) {
                unsized_part = TyKind::Alias(chalk_ir::AliasTy::Projection(ProjectionTy {
                    associated_ty_id: *id,
                    substitution: subst.clone(),
                }))
                .intern(Interner);
            }
            unsized_part = normalize(db, trait_env, unsized_part);
            let metadata = match unsized_part.kind(Interner) {
                TyKind::Slice(_) | TyKind::Str => {
                    scalar_unit(dl, Primitive::Int(dl.ptr_sized_integer(), false))
                }
                TyKind::Dyn(..) => {
                    let mut vtable = scalar_unit(dl, Primitive::Pointer(AddressSpace::DATA));
                    vtable.valid_range_mut().start = 1;
                    vtable
                }
                _ => {
                    // pointee is sized
                    return Ok(Arc::new(Layout::scalar(dl, data_ptr)));
                }
            };

            // Effectively a (ptr, meta) tuple.
            cx.calc.scalar_pair(data_ptr, metadata)
        }
        TyKind::FnDef(_, _) => layout_of_unit(&cx)?,
        TyKind::Never => cx.calc.layout_of_never_type(),
        TyKind::Dyn(_) | TyKind::Foreign(_) => {
            let mut unit = layout_of_unit(&cx)?;
            match &mut unit.backend_repr {
                BackendRepr::Memory { sized } => *sized = false,
                _ => return Err(LayoutError::Unknown),
            }
            unit
        }
        TyKind::Function(_) => {
            let mut ptr = scalar_unit(dl, Primitive::Pointer(dl.instruction_address_space));
            ptr.valid_range_mut().start = 1;
            Layout::scalar(dl, ptr)
        }
        TyKind::OpaqueType(opaque_ty_id, _) => {
            let impl_trait_id = db.lookup_intern_impl_trait_id((*opaque_ty_id).into());
            match impl_trait_id {
                crate::ImplTraitId::ReturnTypeImplTrait(func, idx) => {
                    let infer = db.infer(func.into());
                    return db.layout_of_ty(infer.type_of_rpit[idx].clone(), trait_env);
                }
                crate::ImplTraitId::TypeAliasImplTrait(..) => {
                    return Err(LayoutError::NotImplemented);
                }
                crate::ImplTraitId::AsyncBlockTypeImplTrait(_, _) => {
                    return Err(LayoutError::NotImplemented)
                }
            }
        }
        TyKind::Closure(c, subst) => {
            let InternedClosure(def, _) = db.lookup_intern_closure((*c).into());
            let infer = db.infer(def);
            let (captures, _) = infer.closure_info(c);
            let fields = captures
                .iter()
                .map(|it| {
                    db.layout_of_ty(
                        it.ty.clone().substitute(Interner, ClosureSubst(subst).parent_subst()),
                        trait_env.clone(),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let fields = fields.iter().map(|it| &**it).collect::<Vec<_>>();
            let fields = fields.iter().collect::<IndexVec<_, _>>();
            cx.calc.univariant(&fields, &ReprOptions::default(), StructKind::AlwaysSized)?
        }
        TyKind::Coroutine(_, _) | TyKind::CoroutineWitness(_, _) => {
            return Err(LayoutError::NotImplemented)
        }
        TyKind::Error => return Err(LayoutError::HasErrorType),
        TyKind::AssociatedType(id, subst) => {
            // Try again with `TyKind::Alias` to normalize the associated type.
            let ty = TyKind::Alias(chalk_ir::AliasTy::Projection(ProjectionTy {
                associated_ty_id: *id,
                substitution: subst.clone(),
            }))
            .intern(Interner);
            return db.layout_of_ty(ty, trait_env);
        }
        TyKind::Alias(_)
        | TyKind::Placeholder(_)
        | TyKind::BoundVar(_)
        | TyKind::InferenceVar(_, _) => return Err(LayoutError::HasPlaceholder),
    };
    Ok(Arc::new(result))
}

pub fn layout_of_ty_recover(
    _: &dyn HirDatabase,
    _: &Cycle,
    _: &Ty,
    _: &Arc<TraitEnvironment>,
) -> Result<Arc<Layout>, LayoutError> {
    Err(LayoutError::RecursiveTypeWithoutIndirection)
}

fn layout_of_unit(cx: &LayoutCx<'_>) -> Result<Layout, LayoutError> {
    cx.calc
        .univariant::<RustcFieldIdx, RustcEnumVariantIdx, &&Layout>(
            IndexSlice::empty(),
            &ReprOptions::default(),
            StructKind::AlwaysSized,
        )
        .map_err(Into::into)
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
