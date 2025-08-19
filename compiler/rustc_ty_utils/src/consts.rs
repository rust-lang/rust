use std::iter;

use rustc_abi::{FIRST_VARIANT, VariantIdx};
use rustc_errors::ErrorGuaranteed;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalDefId;
use rustc_middle::mir::interpret::LitToConstInput;
use rustc_middle::query::Providers;
use rustc_middle::thir::visit;
use rustc_middle::thir::visit::Visitor;
use rustc_middle::ty::abstract_const::CastKind;
use rustc_middle::ty::{self, Expr, TyCtxt, TypeVisitableExt};
use rustc_middle::{bug, mir, thir};
use rustc_span::Span;
use tracing::{debug, instrument};

use crate::errors::{GenericConstantTooComplex, GenericConstantTooComplexSub};

/// Destructures array, ADT or tuple constants into the constants
/// of their fields.
fn destructure_const<'tcx>(
    tcx: TyCtxt<'tcx>,
    const_: ty::Const<'tcx>,
) -> ty::DestructuredConst<'tcx> {
    let ty::ConstKind::Value(cv) = const_.kind() else {
        bug!("cannot destructure constant {:?}", const_)
    };

    let branches = cv.valtree.unwrap_branch();

    let (fields, variant) = match cv.ty.kind() {
        ty::Array(inner_ty, _) | ty::Slice(inner_ty) => {
            // construct the consts for the elements of the array/slice
            let field_consts = branches
                .iter()
                .map(|b| ty::Const::new_value(tcx, *b, *inner_ty))
                .collect::<Vec<_>>();
            debug!(?field_consts);

            (field_consts, None)
        }
        ty::Adt(def, _) if def.variants().is_empty() => bug!("unreachable"),
        ty::Adt(def, args) => {
            let (variant_idx, branches) = if def.is_enum() {
                let (head, rest) = branches.split_first().unwrap();
                (VariantIdx::from_u32(head.unwrap_leaf().to_u32()), rest)
            } else {
                (FIRST_VARIANT, branches)
            };
            let fields = &def.variant(variant_idx).fields;
            let mut field_consts = Vec::with_capacity(fields.len());

            for (field, field_valtree) in iter::zip(fields, branches) {
                let field_ty = field.ty(tcx, args);
                let field_const = ty::Const::new_value(tcx, *field_valtree, field_ty);
                field_consts.push(field_const);
            }
            debug!(?field_consts);

            (field_consts, Some(variant_idx))
        }
        ty::Tuple(elem_tys) => {
            let fields = iter::zip(*elem_tys, branches)
                .map(|(elem_ty, elem_valtree)| ty::Const::new_value(tcx, *elem_valtree, elem_ty))
                .collect::<Vec<_>>();

            (fields, None)
        }
        _ => bug!("cannot destructure constant {:?}", const_),
    };

    let fields = tcx.arena.alloc_from_iter(fields);

    ty::DestructuredConst { variant, fields }
}

/// We do not allow all binary operations in abstract consts, so filter disallowed ones.
fn check_binop(op: mir::BinOp) -> bool {
    use mir::BinOp::*;
    match op {
        Add | AddUnchecked | AddWithOverflow | Sub | SubUnchecked | SubWithOverflow | Mul
        | MulUnchecked | MulWithOverflow | Div | Rem | BitXor | BitAnd | BitOr | Shl
        | ShlUnchecked | Shr | ShrUnchecked | Eq | Lt | Le | Ne | Ge | Gt | Cmp => true,
        Offset => false,
    }
}

/// While we currently allow all unary operations, we still want to explicitly guard against
/// future changes here.
fn check_unop(op: mir::UnOp) -> bool {
    use mir::UnOp::*;
    match op {
        Not | Neg | PtrMetadata => true,
    }
}

fn recurse_build<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &thir::Thir<'tcx>,
    node: thir::ExprId,
    root_span: Span,
) -> Result<ty::Const<'tcx>, ErrorGuaranteed> {
    use thir::ExprKind;
    let node = &body.exprs[node];

    let maybe_supported_error = |a| maybe_supported_error(tcx, a, root_span);
    let error = |a| error(tcx, a, root_span);

    Ok(match &node.kind {
        // I dont know if handling of these 3 is correct
        &ExprKind::Scope { value, .. } => recurse_build(tcx, body, value, root_span)?,
        &ExprKind::PlaceTypeAscription { source, .. }
        | &ExprKind::ValueTypeAscription { source, .. } => {
            recurse_build(tcx, body, source, root_span)?
        }
        &ExprKind::PlaceUnwrapUnsafeBinder { .. }
        | &ExprKind::ValueUnwrapUnsafeBinder { .. }
        | &ExprKind::WrapUnsafeBinder { .. } => {
            todo!("FIXME(unsafe_binders)")
        }
        &ExprKind::Literal { lit, neg } => {
            let sp = node.span;
            tcx.at(sp).lit_to_const(LitToConstInput { lit: lit.node, ty: node.ty, neg })
        }
        &ExprKind::NonHirLiteral { lit, user_ty: _ } => {
            let val = ty::ValTree::from_scalar_int(tcx, lit);
            ty::Const::new_value(tcx, val, node.ty)
        }
        &ExprKind::ZstLiteral { user_ty: _ } => ty::Const::zero_sized(tcx, node.ty),
        &ExprKind::NamedConst { def_id, args, user_ty: _ } => {
            let uneval = ty::UnevaluatedConst::new(def_id, args);
            ty::Const::new_unevaluated(tcx, uneval)
        }
        ExprKind::ConstParam { param, .. } => ty::Const::new_param(tcx, *param),

        ExprKind::Call { fun, args, .. } => {
            let fun_ty = body.exprs[*fun].ty;
            let fun = recurse_build(tcx, body, *fun, root_span)?;

            let mut new_args = Vec::<ty::Const<'tcx>>::with_capacity(args.len());
            for &id in args.iter() {
                new_args.push(recurse_build(tcx, body, id, root_span)?);
            }
            ty::Const::new_expr(tcx, Expr::new_call(tcx, fun_ty, fun, new_args))
        }
        &ExprKind::Binary { op, lhs, rhs } if check_binop(op) => {
            let lhs_ty = body.exprs[lhs].ty;
            let lhs = recurse_build(tcx, body, lhs, root_span)?;
            let rhs_ty = body.exprs[rhs].ty;
            let rhs = recurse_build(tcx, body, rhs, root_span)?;
            ty::Const::new_expr(tcx, Expr::new_binop(tcx, op, lhs_ty, rhs_ty, lhs, rhs))
        }
        &ExprKind::Unary { op, arg } if check_unop(op) => {
            let arg_ty = body.exprs[arg].ty;
            let arg = recurse_build(tcx, body, arg, root_span)?;
            ty::Const::new_expr(tcx, Expr::new_unop(tcx, op, arg_ty, arg))
        }
        // This is necessary so that the following compiles:
        //
        // ```
        // fn foo<const N: usize>(a: [(); N + 1]) {
        //     bar::<{ N + 1 }>();
        // }
        // ```
        ExprKind::Block { block } => {
            if let thir::Block { stmts: box [], expr: Some(e), .. } = &body.blocks[*block] {
                recurse_build(tcx, body, *e, root_span)?
            } else {
                maybe_supported_error(GenericConstantTooComplexSub::BlockNotSupported(node.span))?
            }
        }
        // `ExprKind::Use` happens when a `hir::ExprKind::Cast` is a
        // "coercion cast" i.e. using a coercion or is a no-op.
        // This is important so that `N as usize as usize` doesn't unify with `N as usize`. (untested)
        &ExprKind::Use { source } => {
            let value_ty = body.exprs[source].ty;
            let value = recurse_build(tcx, body, source, root_span)?;
            ty::Const::new_expr(tcx, Expr::new_cast(tcx, CastKind::Use, value_ty, value, node.ty))
        }
        &ExprKind::Cast { source } => {
            let value_ty = body.exprs[source].ty;
            let value = recurse_build(tcx, body, source, root_span)?;
            ty::Const::new_expr(tcx, Expr::new_cast(tcx, CastKind::As, value_ty, value, node.ty))
        }
        ExprKind::Borrow { arg, .. } => {
            let arg_node = &body.exprs[*arg];

            // Skip reborrows for now until we allow Deref/Borrow/RawBorrow
            // expressions.
            // FIXME(generic_const_exprs): Verify/explain why this is sound
            if let ExprKind::Deref { arg } = arg_node.kind {
                recurse_build(tcx, body, arg, root_span)?
            } else {
                maybe_supported_error(GenericConstantTooComplexSub::BorrowNotSupported(node.span))?
            }
        }
        // FIXME(generic_const_exprs): We may want to support these.
        ExprKind::RawBorrow { .. } | ExprKind::Deref { .. } => maybe_supported_error(
            GenericConstantTooComplexSub::AddressAndDerefNotSupported(node.span),
        )?,
        ExprKind::Repeat { .. } | ExprKind::Array { .. } => {
            maybe_supported_error(GenericConstantTooComplexSub::ArrayNotSupported(node.span))?
        }
        ExprKind::NeverToAny { .. } => {
            maybe_supported_error(GenericConstantTooComplexSub::NeverToAnyNotSupported(node.span))?
        }
        ExprKind::Tuple { .. } => {
            maybe_supported_error(GenericConstantTooComplexSub::TupleNotSupported(node.span))?
        }
        ExprKind::Index { .. } => {
            maybe_supported_error(GenericConstantTooComplexSub::IndexNotSupported(node.span))?
        }
        ExprKind::Field { .. } => {
            maybe_supported_error(GenericConstantTooComplexSub::FieldNotSupported(node.span))?
        }
        ExprKind::ConstBlock { .. } => {
            maybe_supported_error(GenericConstantTooComplexSub::ConstBlockNotSupported(node.span))?
        }
        ExprKind::Adt(_) => {
            maybe_supported_error(GenericConstantTooComplexSub::AdtNotSupported(node.span))?
        }
        // dont know if this is correct
        ExprKind::PointerCoercion { .. } => {
            error(GenericConstantTooComplexSub::PointerNotSupported(node.span))?
        }
        ExprKind::Yield { .. } => {
            error(GenericConstantTooComplexSub::YieldNotSupported(node.span))?
        }
        ExprKind::Continue { .. }
        | ExprKind::ConstContinue { .. }
        | ExprKind::Break { .. }
        | ExprKind::Loop { .. }
        | ExprKind::LoopMatch { .. } => {
            error(GenericConstantTooComplexSub::LoopNotSupported(node.span))?
        }
        ExprKind::Box { .. } => error(GenericConstantTooComplexSub::BoxNotSupported(node.span))?,
        ExprKind::ByUse { .. } => {
            error(GenericConstantTooComplexSub::ByUseNotSupported(node.span))?
        }
        ExprKind::Unary { .. } => unreachable!(),
        // we handle valid unary/binary ops above
        ExprKind::Binary { .. } => {
            error(GenericConstantTooComplexSub::BinaryNotSupported(node.span))?
        }
        ExprKind::LogicalOp { .. } => {
            error(GenericConstantTooComplexSub::LogicalOpNotSupported(node.span))?
        }
        ExprKind::Assign { .. } | ExprKind::AssignOp { .. } => {
            error(GenericConstantTooComplexSub::AssignNotSupported(node.span))?
        }
        // FIXME(explicit_tail_calls): maybe get `become` a new error
        ExprKind::Closure { .. } | ExprKind::Return { .. } | ExprKind::Become { .. } => {
            error(GenericConstantTooComplexSub::ClosureAndReturnNotSupported(node.span))?
        }
        // let expressions imply control flow
        ExprKind::Match { .. } | ExprKind::If { .. } | ExprKind::Let { .. } => {
            error(GenericConstantTooComplexSub::ControlFlowNotSupported(node.span))?
        }
        ExprKind::InlineAsm { .. } => {
            error(GenericConstantTooComplexSub::InlineAsmNotSupported(node.span))?
        }

        // we dont permit let stmts so `VarRef` and `UpvarRef` cant happen
        ExprKind::VarRef { .. }
        | ExprKind::UpvarRef { .. }
        | ExprKind::StaticRef { .. }
        | ExprKind::OffsetOf { .. }
        | ExprKind::ThreadLocalRef(_) => {
            error(GenericConstantTooComplexSub::OperationNotSupported(node.span))?
        }
    })
}

struct IsThirPolymorphic<'a, 'tcx> {
    is_poly: bool,
    thir: &'a thir::Thir<'tcx>,
}

fn error(
    tcx: TyCtxt<'_>,
    sub: GenericConstantTooComplexSub,
    root_span: Span,
) -> Result<!, ErrorGuaranteed> {
    let reported = tcx.dcx().emit_err(GenericConstantTooComplex {
        span: root_span,
        maybe_supported: false,
        sub,
    });

    Err(reported)
}

fn maybe_supported_error(
    tcx: TyCtxt<'_>,
    sub: GenericConstantTooComplexSub,
    root_span: Span,
) -> Result<!, ErrorGuaranteed> {
    let reported = tcx.dcx().emit_err(GenericConstantTooComplex {
        span: root_span,
        maybe_supported: true,
        sub,
    });

    Err(reported)
}

impl<'a, 'tcx> IsThirPolymorphic<'a, 'tcx> {
    fn expr_is_poly(&mut self, expr: &thir::Expr<'tcx>) -> bool {
        if expr.ty.has_non_region_param() {
            return true;
        }

        match expr.kind {
            thir::ExprKind::NamedConst { args, .. } | thir::ExprKind::ConstBlock { args, .. } => {
                args.has_non_region_param()
            }
            thir::ExprKind::ConstParam { .. } => true,
            thir::ExprKind::Repeat { value, count } => {
                self.visit_expr(&self.thir()[value]);
                count.has_non_region_param()
            }
            thir::ExprKind::Scope { .. }
            | thir::ExprKind::Box { .. }
            | thir::ExprKind::If { .. }
            | thir::ExprKind::Call { .. }
            | thir::ExprKind::ByUse { .. }
            | thir::ExprKind::Deref { .. }
            | thir::ExprKind::Binary { .. }
            | thir::ExprKind::LogicalOp { .. }
            | thir::ExprKind::Unary { .. }
            | thir::ExprKind::Cast { .. }
            | thir::ExprKind::Use { .. }
            | thir::ExprKind::NeverToAny { .. }
            | thir::ExprKind::PointerCoercion { .. }
            | thir::ExprKind::Loop { .. }
            | thir::ExprKind::LoopMatch { .. }
            | thir::ExprKind::Let { .. }
            | thir::ExprKind::Match { .. }
            | thir::ExprKind::Block { .. }
            | thir::ExprKind::Assign { .. }
            | thir::ExprKind::AssignOp { .. }
            | thir::ExprKind::Field { .. }
            | thir::ExprKind::Index { .. }
            | thir::ExprKind::VarRef { .. }
            | thir::ExprKind::UpvarRef { .. }
            | thir::ExprKind::Borrow { .. }
            | thir::ExprKind::RawBorrow { .. }
            | thir::ExprKind::Break { .. }
            | thir::ExprKind::Continue { .. }
            | thir::ExprKind::ConstContinue { .. }
            | thir::ExprKind::Return { .. }
            | thir::ExprKind::Become { .. }
            | thir::ExprKind::Array { .. }
            | thir::ExprKind::Tuple { .. }
            | thir::ExprKind::Adt(_)
            | thir::ExprKind::PlaceTypeAscription { .. }
            | thir::ExprKind::ValueTypeAscription { .. }
            | thir::ExprKind::PlaceUnwrapUnsafeBinder { .. }
            | thir::ExprKind::ValueUnwrapUnsafeBinder { .. }
            | thir::ExprKind::WrapUnsafeBinder { .. }
            | thir::ExprKind::Closure(_)
            | thir::ExprKind::Literal { .. }
            | thir::ExprKind::NonHirLiteral { .. }
            | thir::ExprKind::ZstLiteral { .. }
            | thir::ExprKind::StaticRef { .. }
            | thir::ExprKind::InlineAsm(_)
            | thir::ExprKind::OffsetOf { .. }
            | thir::ExprKind::ThreadLocalRef(_)
            | thir::ExprKind::Yield { .. } => false,
        }
    }
    fn pat_is_poly(&mut self, pat: &thir::Pat<'tcx>) -> bool {
        if pat.ty.has_non_region_param() {
            return true;
        }

        match pat.kind {
            thir::PatKind::Constant { value } => value.has_non_region_param(),
            thir::PatKind::Range(ref range) => {
                let &thir::PatRange { lo, hi, .. } = range.as_ref();
                lo.has_non_region_param() || hi.has_non_region_param()
            }
            _ => false,
        }
    }
}

impl<'a, 'tcx> visit::Visitor<'a, 'tcx> for IsThirPolymorphic<'a, 'tcx> {
    fn thir(&self) -> &'a thir::Thir<'tcx> {
        self.thir
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_expr(&mut self, expr: &'a thir::Expr<'tcx>) {
        self.is_poly |= self.expr_is_poly(expr);
        if !self.is_poly {
            visit::walk_expr(self, expr)
        }
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_pat(&mut self, pat: &'a thir::Pat<'tcx>) {
        self.is_poly |= self.pat_is_poly(pat);
        if !self.is_poly {
            visit::walk_pat(self, pat);
        }
    }
}

/// Builds an abstract const, do not use this directly, but use `AbstractConst::new` instead.
fn thir_abstract_const<'tcx>(
    tcx: TyCtxt<'tcx>,
    def: LocalDefId,
) -> Result<Option<ty::EarlyBinder<'tcx, ty::Const<'tcx>>>, ErrorGuaranteed> {
    if !tcx.features().generic_const_exprs() {
        return Ok(None);
    }

    match tcx.def_kind(def) {
        // FIXME(generic_const_exprs): We currently only do this for anonymous constants,
        // meaning that we do not look into associated constants. I(@lcnr) am not yet sure whether
        // we want to look into them or treat them as opaque projections.
        //
        // Right now we do neither of that and simply always fail to unify them.
        DefKind::AnonConst | DefKind::InlineConst => (),
        _ => return Ok(None),
    }

    let body = tcx.thir_body(def)?;
    let (body, body_id) = (&*body.0.borrow(), body.1);

    let mut is_poly_vis = IsThirPolymorphic { is_poly: false, thir: body };
    visit::walk_expr(&mut is_poly_vis, &body[body_id]);
    if !is_poly_vis.is_poly {
        return Ok(None);
    }

    let root_span = body.exprs[body_id].span;

    Ok(Some(ty::EarlyBinder::bind(recurse_build(tcx, body, body_id, root_span)?)))
}

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers { destructure_const, thir_abstract_const, ..*providers };
}
