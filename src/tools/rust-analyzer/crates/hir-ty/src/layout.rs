//! Compute the binary representation of a type

use std::fmt;

use hir_def::{
    AdtId, LocalFieldId, StructId,
    attrs::AttrFlags,
    layout::{LayoutCalculatorError, LayoutData},
};
use la_arena::{Idx, RawIdx};

use rustc_abi::{
    AddressSpace, Float, Integer, LayoutCalculator, Primitive, ReprOptions, Scalar, StructKind,
    TargetDataLayout, WrappingRange,
};
use rustc_index::IndexVec;
use rustc_type_ir::{
    FloatTy, IntTy, UintTy,
    inherent::{IntoKind, SliceLike},
};
use triomphe::Arc;

use crate::{
    InferenceResult, ParamEnvAndCrate,
    consteval::try_const_usize,
    db::HirDatabase,
    next_solver::{
        DbInterner, GenericArgs, ParamEnv, Ty, TyKind, TypingMode,
        infer::{DbInternerInferExt, traits::ObligationCause},
    },
};

pub(crate) use self::adt::layout_of_adt_cycle_result;
pub use self::{adt::layout_of_adt_query, target::target_data_layout_query};

pub(crate) mod adt;
pub(crate) mod target;

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
fn layout_of_simd_ty<'db>(
    db: &'db dyn HirDatabase,
    id: StructId,
    repr_packed: bool,
    args: &GenericArgs<'db>,
    env: ParamEnvAndCrate<'db>,
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
        .map(|f| (*f.1).instantiate(DbInterner::new_no_crate(db), args).kind())
    else {
        return Err(LayoutError::InvalidSimdType);
    };

    let e_len = try_const_usize(db, e_len).ok_or(LayoutError::HasErrorConst)? as u64;
    let e_ly = db.layout_of_ty(e_ty, env)?;

    let cx = LayoutCx::new(dl);
    Ok(Arc::new(cx.calc.simd_type(e_ly, e_len, repr_packed)?))
}

pub fn layout_of_ty_query<'db>(
    db: &'db dyn HirDatabase,
    ty: Ty<'db>,
    trait_env: ParamEnvAndCrate<'db>,
) -> Result<Arc<Layout>, LayoutError> {
    let krate = trait_env.krate;
    let interner = DbInterner::new_with(db, krate);
    let Ok(target) = db.target_data_layout(krate) else {
        return Err(LayoutError::TargetLayoutNotAvailable);
    };
    let dl = &*target;
    let cx = LayoutCx::new(dl);
    let infer_ctxt = interner.infer_ctxt().build(TypingMode::PostAnalysis);
    let cause = ObligationCause::dummy();
    let ty = infer_ctxt.at(&cause, ParamEnv::empty()).deeply_normalize(ty).unwrap_or(ty);
    let result = match ty.kind() {
        TyKind::Adt(def, args) => {
            match def.inner().id {
                hir_def::AdtId::StructId(s) => {
                    let repr = AttrFlags::repr(db, s.into()).unwrap_or_default();
                    if repr.simd() {
                        return layout_of_simd_ty(db, s, repr.packed(), &args, trait_env, &target);
                    }
                }
                _ => {}
            }
            return db.layout_of_adt(def.inner().id, args, trait_env);
        }
        TyKind::Bool => Layout::scalar(
            dl,
            Scalar::Initialized {
                value: Primitive::Int(Integer::I8, false),
                valid_range: WrappingRange { start: 0, end: 1 },
            },
        ),
        TyKind::Char => Layout::scalar(
            dl,
            Scalar::Initialized {
                value: Primitive::Int(Integer::I32, false),
                valid_range: WrappingRange { start: 0, end: 0x10FFFF },
            },
        ),
        TyKind::Int(i) => Layout::scalar(
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
        TyKind::Uint(i) => Layout::scalar(
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
        TyKind::Float(f) => Layout::scalar(
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
        TyKind::Tuple(tys) => {
            let kind =
                if tys.len() == 0 { StructKind::AlwaysSized } else { StructKind::MaybeUnsized };

            let fields =
                tys.iter().map(|k| db.layout_of_ty(k, trait_env)).collect::<Result<Vec<_>, _>>()?;
            let fields = fields.iter().map(|it| &**it).collect::<Vec<_>>();
            let fields = fields.iter().collect::<IndexVec<_, _>>();
            cx.calc.univariant(&fields, &ReprOptions::default(), kind)?
        }
        TyKind::Array(element, count) => {
            let count = try_const_usize(db, count).ok_or(LayoutError::HasErrorConst)? as u64;
            let element = db.layout_of_ty(element, trait_env)?;
            cx.calc.array_like::<_, _, ()>(&element, Some(count))?
        }
        TyKind::Slice(element) => {
            let element = db.layout_of_ty(element, trait_env)?;
            cx.calc.array_like::<_, _, ()>(&element, None)?
        }
        TyKind::Str => {
            let element = scalar_unit(dl, Primitive::Int(Integer::I8, false));
            cx.calc.array_like::<_, _, ()>(&Layout::scalar(dl, element), None)?
        }
        // Potentially-wide pointers.
        TyKind::Ref(_, pointee, _) | TyKind::RawPtr(pointee, _) => {
            let mut data_ptr = scalar_unit(dl, Primitive::Pointer(AddressSpace::ZERO));
            if matches!(ty.kind(), TyKind::Ref(..)) {
                data_ptr.valid_range_mut().start = 1;
            }

            // FIXME(next-solver)
            // let pointee = tcx.normalize_erasing_regions(param_env, pointee);
            // if pointee.is_sized(tcx.at(DUMMY_SP), param_env) {
            //     return Ok(tcx.mk_layout(LayoutS::scalar(cx, data_ptr)));
            // }

            let unsized_part = struct_tail_erasing_lifetimes(db, pointee);
            // FIXME(next-solver)
            /*
            if let TyKind::AssociatedType(id, subst) = unsized_part.kind(Interner) {
                unsized_part = TyKind::Alias(chalk_ir::AliasTy::Projection(ProjectionTy {
                    associated_ty_id: *id,
                    substitution: subst.clone(),
                }))
                .intern(Interner);
            }
            unsized_part = normalize(db, trait_env, unsized_part);
            */
            let metadata = match unsized_part.kind() {
                TyKind::Slice(_) | TyKind::Str => {
                    scalar_unit(dl, Primitive::Int(dl.ptr_sized_integer(), false))
                }
                TyKind::Dynamic(..) => {
                    let mut vtable = scalar_unit(dl, Primitive::Pointer(AddressSpace::ZERO));
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
        TyKind::FnDef(..) => LayoutData::unit(dl, true),
        TyKind::Dynamic(..) | TyKind::Foreign(_) => LayoutData::unit(dl, false),
        TyKind::FnPtr(..) => {
            let mut ptr = scalar_unit(dl, Primitive::Pointer(dl.instruction_address_space));
            ptr.valid_range_mut().start = 1;
            Layout::scalar(dl, ptr)
        }
        TyKind::Closure(id, args) => {
            let def = db.lookup_intern_closure(id.0);
            let infer = InferenceResult::for_body(db, def.0);
            let (captures, _) = infer.closure_info(id.0);
            let fields = captures
                .iter()
                .map(|it| {
                    let ty =
                        it.ty.instantiate(interner, args.split_closure_args_untupled().parent_args);
                    db.layout_of_ty(ty, trait_env)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let fields = fields.iter().map(|it| &**it).collect::<Vec<_>>();
            let fields = fields.iter().collect::<IndexVec<_, _>>();
            cx.calc.univariant(&fields, &ReprOptions::default(), StructKind::AlwaysSized)?
        }

        TyKind::Coroutine(_, _)
        | TyKind::CoroutineWitness(_, _)
        | TyKind::CoroutineClosure(_, _) => {
            return Err(LayoutError::NotImplemented);
        }

        TyKind::Pat(_, _) | TyKind::UnsafeBinder(_) => {
            return Err(LayoutError::NotImplemented);
        }

        TyKind::Error(_) => return Err(LayoutError::HasErrorType),
        TyKind::Placeholder(_)
        | TyKind::Bound(..)
        | TyKind::Infer(..)
        | TyKind::Param(..)
        | TyKind::Alias(..) => {
            return Err(LayoutError::HasPlaceholder);
        }
    };
    Ok(Arc::new(result))
}

pub(crate) fn layout_of_ty_cycle_result<'db>(
    _: &dyn HirDatabase,
    _: Ty<'db>,
    _: ParamEnvAndCrate<'db>,
) -> Result<Arc<Layout>, LayoutError> {
    Err(LayoutError::RecursiveTypeWithoutIndirection)
}

fn struct_tail_erasing_lifetimes<'a>(db: &'a dyn HirDatabase, pointee: Ty<'a>) -> Ty<'a> {
    match pointee.kind() {
        TyKind::Adt(def, args) => {
            let struct_id = match def.inner().id {
                AdtId::StructId(id) => id,
                _ => return pointee,
            };
            let data = struct_id.fields(db);
            let mut it = data.fields().iter().rev();
            match it.next() {
                Some((f, _)) => {
                    let last_field_ty = field_ty(db, struct_id.into(), f, &args);
                    struct_tail_erasing_lifetimes(db, last_field_ty)
                }
                None => pointee,
            }
        }
        TyKind::Tuple(tys) => {
            if let Some(last_field_ty) = tys.iter().next_back() {
                struct_tail_erasing_lifetimes(db, last_field_ty)
            } else {
                pointee
            }
        }
        _ => pointee,
    }
}

fn field_ty<'a>(
    db: &'a dyn HirDatabase,
    def: hir_def::VariantId,
    fd: LocalFieldId,
    args: &GenericArgs<'a>,
) -> Ty<'a> {
    db.field_types(def)[fd].instantiate(DbInterner::new_no_crate(db), args)
}

fn scalar_unit(dl: &TargetDataLayout, value: Primitive) -> Scalar {
    Scalar::Initialized { value, valid_range: WrappingRange::full(value.size(dl)) }
}

#[cfg(test)]
mod tests;
