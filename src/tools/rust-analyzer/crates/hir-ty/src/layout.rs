//! Compute the binary representation of a type

use std::fmt;

use chalk_ir::{AdtId, FloatTy, IntTy, TyKind, UintTy};
use hir_def::{
    LocalFieldId, StructId,
    layout::{
        Float, Integer, LayoutCalculator, LayoutCalculatorError, LayoutData, Primitive,
        ReprOptions, Scalar, StructKind, TargetDataLayout, WrappingRange,
    },
};
use la_arena::{Idx, RawIdx};
use rustc_abi::AddressSpace;
use rustc_index::IndexVec;

use triomphe::Arc;

use crate::{
    Interner, ProjectionTy, Substitution, TraitEnvironment, Ty,
    consteval::try_const_usize,
    db::{HirDatabase, InternedClosure},
    infer::normalize,
    utils::ClosureSubst,
};

pub(crate) use self::adt::layout_of_adt_cycle_result;
pub use self::{adt::layout_of_adt_query, target::target_data_layout_query};

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

fn layout_of_simd_ty(
    db: &dyn HirDatabase,
    id: StructId,
    repr_packed: bool,
    subst: &Substitution,
    env: Arc<TraitEnvironment>,
    dl: &TargetDataLayout,
) -> Result<Arc<Layout>, LayoutError> {
    // Supported SIMD vectors are homogeneous ADTs with exactly one array field:
    //
    // * #[repr(simd)] struct S([T; 4])
    //
    // where T is a primitive scalar (integer/float/pointer).
    let fields = db.field_types(id.into());
    let mut fields = fields.iter();
    let Some(TyKind::Array(e_ty, e_len)) = fields
        .next()
        .filter(|_| fields.next().is_none())
        .map(|f| f.1.clone().substitute(Interner, subst).kind(Interner).clone())
    else {
        return Err(LayoutError::InvalidSimdType);
    };

    let e_len = try_const_usize(db, &e_len).ok_or(LayoutError::HasErrorConst)? as u64;
    let e_ly = db.layout_of_ty(e_ty, env)?;

    let cx = LayoutCx::new(dl);
    Ok(Arc::new(cx.calc.simd_type(e_ly, e_len, repr_packed)?))
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
    let kind = ty.kind(Interner);
    let result = match kind {
        TyKind::Adt(AdtId(def), subst) => {
            if let hir_def::AdtId::StructId(s) = def {
                let data = db.struct_signature(*s);
                let repr = data.repr.unwrap_or_default();
                if repr.simd() {
                    return layout_of_simd_ty(db, *s, repr.packed(), subst, trait_env, &target);
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
            chalk_ir::Scalar::Int(i) => Layout::scalar(
                dl,
                scalar_unit(
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
            ),
            chalk_ir::Scalar::Uint(i) => Layout::scalar(
                dl,
                scalar_unit(
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
            ),
            chalk_ir::Scalar::Float(f) => Layout::scalar(
                dl,
                scalar_unit(
                    dl,
                    Primitive::Float(match f {
                        FloatTy::F16 => Float::F16,
                        FloatTy::F32 => Float::F32,
                        FloatTy::F64 => Float::F64,
                        FloatTy::F128 => Float::F128,
                    }),
                ),
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
            cx.calc.array_like::<_, _, ()>(&element, Some(count))?
        }
        TyKind::Slice(element) => {
            let element = db.layout_of_ty(element.clone(), trait_env)?;
            cx.calc.array_like::<_, _, ()>(&element, None)?
        }
        TyKind::Str => {
            let element = scalar_unit(dl, Primitive::Int(Integer::I8, false));
            cx.calc.array_like::<_, _, ()>(&Layout::scalar(dl, element), None)?
        }
        // Potentially-wide pointers.
        TyKind::Ref(_, _, pointee) | TyKind::Raw(_, pointee) => {
            let mut data_ptr = scalar_unit(dl, Primitive::Pointer(AddressSpace::DATA));
            if matches!(ty.kind(Interner), TyKind::Ref(..)) {
                data_ptr.valid_range_mut().start = 1;
            }

            // let pointee = tcx.normalize_erasing_regions(param_env, pointee);
            // if pointee.is_sized(tcx.at(DUMMY_SP), param_env) {
            //     return Ok(tcx.mk_layout(LayoutData::scalar(cx, data_ptr)));
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
            LayoutData::scalar_pair(dl, data_ptr, metadata)
        }
        TyKind::Never => LayoutData::never_type(dl),
        TyKind::FnDef(..) | TyKind::Dyn(_) | TyKind::Foreign(_) => {
            let sized = matches!(kind, TyKind::FnDef(..));
            LayoutData::unit(dl, sized)
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
                    return Err(LayoutError::NotImplemented);
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
            return Err(LayoutError::NotImplemented);
        }
        TyKind::Error => return Err(LayoutError::HasErrorType),
        TyKind::AssociatedType(id, subst) => {
            // Try again with `TyKind::Alias` to normalize the associated type.
            // Usually we should not try to normalize `TyKind::AssociatedType`, but layout calculation is used
            // in monomorphized MIR where this is okay. If outside monomorphization, this will lead to cycle,
            // which we will recover from with an error.
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

pub(crate) fn layout_of_ty_cycle_result(
    _: &dyn HirDatabase,
    _: Ty,
    _: Arc<TraitEnvironment>,
) -> Result<Arc<Layout>, LayoutError> {
    Err(LayoutError::RecursiveTypeWithoutIndirection)
}

fn struct_tail_erasing_lifetimes(db: &dyn HirDatabase, pointee: Ty) -> Ty {
    match pointee.kind(Interner) {
        &TyKind::Adt(AdtId(hir_def::AdtId::StructId(i)), ref subst) => {
            let data = i.fields(db);
            let mut it = data.fields().iter().rev();
            match it.next() {
                Some((f, _)) => {
                    let last_field_ty = field_ty(db, i.into(), f, subst);
                    struct_tail_erasing_lifetimes(db, last_field_ty)
                }
                None => pointee,
            }
        }
        TyKind::Tuple(_, subst) => {
            if let Some(last_field_ty) =
                subst.iter(Interner).last().and_then(|arg| arg.ty(Interner))
            {
                struct_tail_erasing_lifetimes(db, last_field_ty.clone())
            } else {
                pointee
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

#[cfg(test)]
mod tests;
