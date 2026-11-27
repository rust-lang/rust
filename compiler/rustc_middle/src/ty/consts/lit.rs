use rustc_abi::Size;
use rustc_apfloat::Float;
use rustc_apfloat::ieee::{Double, Half, Quad, Single};
use rustc_ast::{LitFloatType, LitIntType, LitKind};
use rustc_attr_ir::lang_items::LangItem;
use rustc_macros::StableHash;
use rustc_span::{Symbol, bug};
use tracing::{instrument, trace};

use crate::mir;
use crate::mir::interpret::{CTFE_ALLOC_SALT, Scalar};
use crate::ty::{self, ScalarInt, Ty, TyCtxt, TypeVisitableExt as _};

/// Input argument for `tcx.lit_to_const`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, StableHash)]
pub struct LitToConstInput<'tcx> {
    /// The absolute value of the resultant constant.
    pub lit: LitKind,
    /// The type of the constant.
    ///
    /// `None` is used by const generics when the type of the constant is unknown, e.g.
    /// if there are inference variables
    pub ty: Option<Ty<'tcx>>,
    /// If the constant is negative.
    pub neg: bool,
}

/// Checks whether a literal can be interpreted as a const of the given type.
pub fn const_lit_matches_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    kind: &LitKind,
    ty: Ty<'tcx>,
    neg: bool,
) -> bool {
    match (*kind, ty.kind()) {
        (LitKind::Str(..), ty::Ref(_, inner_ty, _)) if inner_ty.is_str() => true,
        (LitKind::Str(..), ty::Str) if tcx.features().deref_patterns() => true,
        (LitKind::ByteStr(..), ty::Ref(_, inner_ty, _))
            if let ty::Slice(ty) | ty::Array(ty, _) = inner_ty.kind()
                && matches!(ty.kind(), ty::Uint(ty::UintTy::U8)) =>
        {
            true
        }
        (LitKind::ByteStr(..), ty::Slice(inner_ty) | ty::Array(inner_ty, _))
            if tcx.features().deref_patterns()
                && matches!(inner_ty.kind(), ty::Uint(ty::UintTy::U8)) =>
        {
            true
        }
        (LitKind::Byte(..), ty::Uint(ty::UintTy::U8)) => true,
        (LitKind::CStr(..), ty::Ref(_, inner_ty, _))
            if matches!(inner_ty.kind(), ty::Adt(def, _)
                if tcx.is_lang_item(def.did(), LangItem::CStr)) =>
        {
            true
        }
        (LitKind::Int(_, LitIntType::Unsigned(lit_ty)), ty::Uint(expect_ty)) if !neg => {
            lit_ty == *expect_ty
        }
        (LitKind::Int(_, LitIntType::Signed(lit_ty)), ty::Int(expect_ty)) => lit_ty == *expect_ty,
        (LitKind::Int(_, LitIntType::Unsuffixed), ty::Uint(_)) if !neg => true,
        (LitKind::Int(_, LitIntType::Unsuffixed), ty::Int(_)) => true,
        (LitKind::Bool(..), ty::Bool) => true,
        (LitKind::Float(_, LitFloatType::Suffixed(lit_ty)), ty::Float(expect_ty)) => {
            lit_ty == *expect_ty
        }
        (LitKind::Float(_, LitFloatType::Unsuffixed), ty::Float(_)) => true,
        (LitKind::Char(..), ty::Char) => true,
        (LitKind::Err(..), _) => true,
        _ => false,
    }
}

#[instrument(skip(tcx, lit_input))]
pub fn lit_to_mir_constant<'tcx>(
    tcx: TyCtxt<'tcx>,
    lit_input: LitToConstInput<'tcx>,
) -> mir::Const<'tcx> {
    let LitToConstInput { lit, ty, neg } = lit_input;

    let ty = ty.expect("type of literal must be known at this point");

    if let Err(guar) = ty.error_reported() {
        return mir::Const::Ty(Ty::new_error(tcx, guar), ty::Const::new_error(tcx, guar));
    }

    let lit_ty = match *ty.kind() {
        ty::Pat(base, _) => base,
        _ => ty,
    };

    let trunc = |n| {
        let width = lit_ty.primitive_size(tcx);
        trace!("trunc {} with size {} and shift {}", n, width.bits(), 128 - width.bits());
        let result = width.truncate(n);
        trace!("trunc result: {}", result);
        mir::ConstValue::Scalar(Scalar::from_uint(result, width))
    };

    let value = match (lit, lit_ty.kind()) {
        (LitKind::Str(s, _), ty::Ref(_, inner_ty, _)) if inner_ty.is_str() => {
            let s = s.as_str().as_bytes();
            let len = s.len();
            let allocation = tcx.allocate_bytes_dedup(s, CTFE_ALLOC_SALT);
            mir::ConstValue::Slice { alloc_id: allocation, meta: len.try_into().unwrap() }
        }
        (LitKind::ByteStr(byte_sym, _), ty::Ref(_, inner_ty, _))
            if matches!(inner_ty.kind(), ty::Slice(_)) =>
        {
            let data = byte_sym.as_byte_str();
            let len = data.len();
            let allocation = tcx.allocate_bytes_dedup(data, CTFE_ALLOC_SALT);
            mir::ConstValue::Slice { alloc_id: allocation, meta: len.try_into().unwrap() }
        }
        (LitKind::ByteStr(byte_sym, _), ty::Ref(_, inner_ty, _)) if inner_ty.is_array() => {
            let id = tcx.allocate_bytes_dedup(byte_sym.as_byte_str(), CTFE_ALLOC_SALT);
            mir::ConstValue::Scalar(Scalar::from_pointer(id.into(), &tcx))
        }
        (LitKind::CStr(byte_sym, _), ty::Ref(_, inner_ty, _))
            if let ty::Adt(def, _) = inner_ty.kind()
                && tcx.is_lang_item(def.did(), LangItem::CStr) =>
        {
            let data = byte_sym.as_byte_str();
            let len = data.len();
            let allocation = tcx.allocate_bytes_dedup(data, CTFE_ALLOC_SALT);
            mir::ConstValue::Slice { alloc_id: allocation, meta: len.try_into().unwrap() }
        }
        (LitKind::Byte(n), ty::Uint(ty::UintTy::U8)) => {
            mir::ConstValue::Scalar(Scalar::from_uint(n, Size::from_bytes(1)))
        }
        // Only reachable through a user `impl Neg` for an unsigned type, which violates the
        // orphan rules and has already been reported.
        (LitKind::Int(..), ty::Uint(_)) if neg => {
            let guar = tcx.dcx().delayed_bug("negated unsigned literal without a prior error");
            return mir::Const::Ty(Ty::new_error(tcx, guar), ty::Const::new_error(tcx, guar));
        }
        (LitKind::Int(n, _), ty::Uint(_)) => trunc(n.get()),
        (LitKind::Int(n, _), ty::Int(_)) => {
            // Unsigned "negation" has the same bitwise effect as signed negation,
            // which gets the result we want without additional casts.
            trunc(if neg { u128::wrapping_neg(n.get()) } else { n.get() })
        }
        (LitKind::Float(n, _), ty::Float(fty)) => parse_float_into_constval(n, *fty, neg).unwrap(),
        (LitKind::Bool(b), ty::Bool) => mir::ConstValue::Scalar(Scalar::from_bool(b)),
        (LitKind::Char(c), ty::Char) => mir::ConstValue::Scalar(Scalar::from_char(c)),
        (LitKind::Err(guar), _) => {
            return mir::Const::Ty(Ty::new_error(tcx, guar), ty::Const::new_error(tcx, guar));
        }
        _ => bug!("invalid lit/ty combination in `lit_to_mir_constant`: {lit:?}: {ty:?}"),
    };

    mir::Const::Val(value, ty)
}

fn parse_float_into_constval(
    num: Symbol,
    float_ty: ty::FloatTy,
    neg: bool,
) -> Option<mir::ConstValue> {
    parse_float_into_scalar(num, float_ty, neg).map(|s| mir::ConstValue::Scalar(s.into()))
}

pub fn parse_float_into_scalar(num: Symbol, float_ty: ty::FloatTy, neg: bool) -> Option<ScalarInt> {
    let num = num.as_str();
    match float_ty {
        // FIXME(f16_f128): When available, compare to the library parser as with `f32` and `f64`
        ty::FloatTy::F16 => {
            let mut f = num.parse::<Half>().ok()?;
            if neg {
                f = -f;
            }
            Some(ScalarInt::from(f))
        }
        ty::FloatTy::F32 => {
            let Ok(rust_f) = num.parse::<f32>() else { return None };
            let mut f = num
                .parse::<Single>()
                .unwrap_or_else(|e| panic!("apfloat::ieee::Single failed to parse `{num}`: {e:?}"));

            assert!(
                u128::from(rust_f.to_bits()) == f.to_bits(),
                "apfloat::ieee::Single gave different result for `{}`: \
                 {}({:#x}) vs Rust's {}({:#x})",
                rust_f,
                f,
                f.to_bits(),
                Single::from_bits(rust_f.to_bits().into()),
                rust_f.to_bits()
            );

            if neg {
                f = -f;
            }

            Some(ScalarInt::from(f))
        }
        ty::FloatTy::F64 => {
            let Ok(rust_f) = num.parse::<f64>() else { return None };
            let mut f = num
                .parse::<Double>()
                .unwrap_or_else(|e| panic!("apfloat::ieee::Double failed to parse `{num}`: {e:?}"));

            assert!(
                u128::from(rust_f.to_bits()) == f.to_bits(),
                "apfloat::ieee::Double gave different result for `{}`: \
                 {}({:#x}) vs Rust's {}({:#x})",
                rust_f,
                f,
                f.to_bits(),
                Double::from_bits(rust_f.to_bits().into()),
                rust_f.to_bits()
            );

            if neg {
                f = -f;
            }

            Some(ScalarInt::from(f))
        }
        // FIXME(f16_f128): When available, compare to the library parser as with `f32` and `f64`
        ty::FloatTy::F128 => {
            let mut f = num.parse::<Quad>().ok()?;
            if neg {
                f = -f;
            }
            Some(ScalarInt::from(f))
        }
    }
}
