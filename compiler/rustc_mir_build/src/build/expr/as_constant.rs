//! See docs in build/expr/mod.rs

use crate::build::{parse_float_into_constval, Builder};
use rustc_ast as ast;
use rustc_middle::mir;
use rustc_middle::mir::interpret::{Allocation, LitToConstError, LitToConstInput, Scalar};
use rustc_middle::mir::*;
use rustc_middle::thir::*;
use rustc_middle::ty::{
    self, CanonicalUserType, CanonicalUserTypeAnnotation, TyCtxt, UserTypeAnnotationIndex,
};
use rustc_span::DUMMY_SP;
use rustc_target::abi::Size;

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Compile `expr`, yielding a compile-time constant. Assumes that
    /// `expr` is a valid compile-time constant!
    pub(crate) fn as_constant(&mut self, expr: &Expr<'tcx>) -> ConstOperand<'tcx> {
        let this = self;
        let tcx = this.tcx;
        let Expr { ty, temp_lifetime: _, span, ref kind } = *expr;
        match kind {
            ExprKind::Scope { region_scope: _, lint_level: _, value } => {
                this.as_constant(&this.thir[*value])
            }
            _ => as_constant_inner(
                expr,
                |user_ty| {
                    Some(this.canonical_user_type_annotations.push(CanonicalUserTypeAnnotation {
                        span,
                        user_ty: user_ty.clone(),
                        inferred_ty: ty,
                    }))
                },
                tcx,
            ),
        }
    }
}

pub fn as_constant_inner<'tcx>(
    expr: &Expr<'tcx>,
    push_cuta: impl FnMut(&Box<CanonicalUserType<'tcx>>) -> Option<UserTypeAnnotationIndex>,
    tcx: TyCtxt<'tcx>,
) -> ConstOperand<'tcx> {
    let Expr { ty, temp_lifetime: _, span, ref kind } = *expr;
    match *kind {
        ExprKind::Literal { lit, neg } => {
            let const_ = match lit_to_mir_constant(tcx, LitToConstInput { lit: &lit.node, ty, neg })
            {
                Ok(c) => c,
                Err(LitToConstError::Reported(guar)) => {
                    Const::Ty(ty::Const::new_error(tcx, guar, ty))
                }
                Err(LitToConstError::TypeError) => {
                    bug!("encountered type error in `lit_to_mir_constant`")
                }
            };

            ConstOperand { span, user_ty: None, const_ }
        }
        ExprKind::NonHirLiteral { lit, ref user_ty } => {
            let user_ty = user_ty.as_ref().and_then(push_cuta);

            let const_ = Const::Val(ConstValue::Scalar(Scalar::Int(lit)), ty);

            ConstOperand { span, user_ty, const_ }
        }
        ExprKind::ZstLiteral { ref user_ty } => {
            let user_ty = user_ty.as_ref().and_then(push_cuta);

            let const_ = Const::Val(ConstValue::ZeroSized, ty);

            ConstOperand { span, user_ty, const_ }
        }
        ExprKind::NamedConst { def_id, args, ref user_ty } => {
            let user_ty = user_ty.as_ref().and_then(push_cuta);

            let uneval = mir::UnevaluatedConst::new(def_id, args);
            let const_ = Const::Unevaluated(uneval, ty);

            ConstOperand { user_ty, span, const_ }
        }
        ExprKind::ConstParam { param, def_id: _ } => {
            let const_param = ty::Const::new_param(tcx, param, expr.ty);
            let const_ = Const::Ty(const_param);

            ConstOperand { user_ty: None, span, const_ }
        }
        ExprKind::ConstBlock { did: def_id, args } => {
            let uneval = mir::UnevaluatedConst::new(def_id, args);
            let const_ = Const::Unevaluated(uneval, ty);

            ConstOperand { user_ty: None, span, const_ }
        }
        ExprKind::StaticRef { alloc_id, ty, .. } => {
            let const_val = ConstValue::Scalar(Scalar::from_pointer(alloc_id.into(), &tcx));
            let const_ = Const::Val(const_val, ty);

            ConstOperand { span, user_ty: None, const_ }
        }
        _ => span_bug!(span, "expression is not a valid constant {:?}", kind),
    }
}

#[instrument(skip(tcx, lit_input))]
fn lit_to_mir_constant<'tcx>(
    tcx: TyCtxt<'tcx>,
    lit_input: LitToConstInput<'tcx>,
) -> Result<Const<'tcx>, LitToConstError> {
    let LitToConstInput { lit, ty, neg } = lit_input;
    let trunc = |n| {
        let param_ty = ty::ParamEnv::reveal_all().and(ty);
        let width = tcx
            .layout_of(param_ty)
            .map_err(|_| {
                LitToConstError::Reported(tcx.sess.delay_span_bug(
                    DUMMY_SP,
                    format!("couldn't compute width of literal: {:?}", lit_input.lit),
                ))
            })?
            .size;
        trace!("trunc {} with size {} and shift {}", n, width.bits(), 128 - width.bits());
        let result = width.truncate(n);
        trace!("trunc result: {}", result);
        Ok(ConstValue::Scalar(Scalar::from_uint(result, width)))
    };

    let value = match (lit, &ty.kind()) {
        (ast::LitKind::Str(s, _), ty::Ref(_, inner_ty, _)) if inner_ty.is_str() => {
            let s = s.as_str();
            let allocation = Allocation::from_bytes_byte_aligned_immutable(s.as_bytes());
            let allocation = tcx.mk_const_alloc(allocation);
            ConstValue::Slice { data: allocation, meta: allocation.inner().size().bytes() }
        }
        (ast::LitKind::ByteStr(data, _), ty::Ref(_, inner_ty, _))
            if matches!(inner_ty.kind(), ty::Slice(_)) =>
        {
            let allocation = Allocation::from_bytes_byte_aligned_immutable(data as &[u8]);
            let allocation = tcx.mk_const_alloc(allocation);
            ConstValue::Slice { data: allocation, meta: allocation.inner().size().bytes() }
        }
        (ast::LitKind::ByteStr(data, _), ty::Ref(_, inner_ty, _)) if inner_ty.is_array() => {
            let id = tcx.allocate_bytes(data);
            ConstValue::Scalar(Scalar::from_pointer(id.into(), &tcx))
        }
        (ast::LitKind::CStr(data, _), ty::Ref(_, inner_ty, _)) if matches!(inner_ty.kind(), ty::Adt(def, _) if Some(def.did()) == tcx.lang_items().c_str()) =>
        {
            let allocation = Allocation::from_bytes_byte_aligned_immutable(data as &[u8]);
            let allocation = tcx.mk_const_alloc(allocation);
            ConstValue::Slice { data: allocation, meta: allocation.inner().size().bytes() }
        }
        (ast::LitKind::Byte(n), ty::Uint(ty::UintTy::U8)) => {
            ConstValue::Scalar(Scalar::from_uint(*n, Size::from_bytes(1)))
        }
        (ast::LitKind::Int(n, _), ty::Uint(_)) | (ast::LitKind::Int(n, _), ty::Int(_)) => {
            trunc(if neg { (*n as i128).overflowing_neg().0 as u128 } else { *n })?
        }
        (ast::LitKind::Float(n, _), ty::Float(fty)) => parse_float_into_constval(*n, *fty, neg)
            .ok_or_else(|| {
                LitToConstError::Reported(tcx.sess.delay_span_bug(
                    DUMMY_SP,
                    format!("couldn't parse float literal: {:?}", lit_input.lit),
                ))
            })?,
        (ast::LitKind::Bool(b), ty::Bool) => ConstValue::Scalar(Scalar::from_bool(*b)),
        (ast::LitKind::Char(c), ty::Char) => ConstValue::Scalar(Scalar::from_char(*c)),
        (ast::LitKind::Err, _) => {
            return Err(LitToConstError::Reported(
                tcx.sess.delay_span_bug(DUMMY_SP, "encountered LitKind::Err during mir build"),
            ));
        }
        _ => return Err(LitToConstError::TypeError),
    };

    Ok(Const::Val(value, ty))
}
