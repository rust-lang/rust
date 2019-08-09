use syntax::ast;
use rustc::ty::{self, Ty, TyCtxt, ParamEnv, layout::Size};
use syntax_pos::symbol::Symbol;
use rustc::mir::interpret::{ConstValue, Scalar};

#[derive(PartialEq)]
crate enum LitToConstError {
    UnparseableFloat,
    Reported,
}

crate fn lit_to_const<'tcx>(
    lit: &'tcx ast::LitKind,
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    neg: bool,
) -> Result<&'tcx ty::Const<'tcx>, LitToConstError> {
    use syntax::ast::*;

    let trunc = |n| {
        let param_ty = ParamEnv::reveal_all().and(ty);
        let width = tcx.layout_of(param_ty).map_err(|_| LitToConstError::Reported)?.size;
        trace!("trunc {} with size {} and shift {}", n, width.bits(), 128 - width.bits());
        let result = truncate(n, width);
        trace!("trunc result: {}", result);
        Ok(ConstValue::Scalar(Scalar::from_uint(result, width)))
    };

    use rustc::mir::interpret::*;
    let lit = match *lit {
        LitKind::Str(ref s, _) => {
            let s = s.as_str();
            let allocation = Allocation::from_byte_aligned_bytes(s.as_bytes());
            let allocation = tcx.intern_const_alloc(allocation);
            ConstValue::Slice { data: allocation, start: 0, end: s.len() }
        },
        LitKind::ByteStr(ref data) => {
            let id = tcx.allocate_bytes(data);
            ConstValue::Scalar(Scalar::Ptr(id.into()))
        },
        LitKind::Byte(n) => ConstValue::Scalar(Scalar::from_uint(n, Size::from_bytes(1))),
        LitKind::Int(n, _) if neg => {
            let n = n as i128;
            let n = n.overflowing_neg().0;
            trunc(n as u128)?
        },
        LitKind::Int(n, _) => trunc(n)?,
        LitKind::Float(n, fty) => {
            parse_float(n, fty, neg).map_err(|_| LitToConstError::UnparseableFloat)?
        }
        LitKind::FloatUnsuffixed(n) => {
            let fty = match ty.sty {
                ty::Float(fty) => fty,
                _ => bug!()
            };
            parse_float(n, fty, neg).map_err(|_| LitToConstError::UnparseableFloat)?
        }
        LitKind::Bool(b) => ConstValue::Scalar(Scalar::from_bool(b)),
        LitKind::Char(c) => ConstValue::Scalar(Scalar::from_char(c)),
        LitKind::Err(_) => unreachable!(),
    };
    Ok(tcx.mk_const(ty::Const { val: lit, ty }))
}

fn parse_float<'tcx>(
    num: Symbol,
    fty: ast::FloatTy,
    neg: bool,
) -> Result<ConstValue<'tcx>, ()> {
    let num = num.as_str();
    use rustc_apfloat::ieee::{Single, Double};
    let scalar = match fty {
        ast::FloatTy::F32 => {
            num.parse::<f32>().map_err(|_| ())?;
            let mut f = num.parse::<Single>().unwrap_or_else(|e| {
                panic!("apfloat::ieee::Single failed to parse `{}`: {:?}", num, e)
            });
            if neg {
                f = -f;
            }
            Scalar::from_f32(f)
        }
        ast::FloatTy::F64 => {
            num.parse::<f64>().map_err(|_| ())?;
            let mut f = num.parse::<Double>().unwrap_or_else(|e| {
                panic!("apfloat::ieee::Double failed to parse `{}`: {:?}", num, e)
            });
            if neg {
                f = -f;
            }
            Scalar::from_f64(f)
        }
    };

    Ok(ConstValue::Scalar(scalar))
}
