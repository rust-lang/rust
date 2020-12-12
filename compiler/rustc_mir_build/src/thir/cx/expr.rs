use crate::thir::cx::block;
use crate::thir::cx::to_ref::ToRef;
use crate::thir::cx::Cx;
use crate::thir::util::UserAnnotatedTyHelpers;
use crate::thir::*;
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, CtorOf, DefKind, Res};
use rustc_index::vec::Idx;
use rustc_middle::hir::place::PlaceBase as HirPlaceBase;
use rustc_middle::hir::place::ProjectionKind as HirProjectionKind;
use rustc_middle::mir::interpret::Scalar;
use rustc_middle::mir::BorrowKind;
use rustc_middle::ty::adjustment::{
    Adjust, Adjustment, AutoBorrow, AutoBorrowMutability, PointerCast,
};
use rustc_middle::ty::subst::{InternalSubsts, SubstsRef};
use rustc_middle::ty::{self, AdtKind, Ty};
use rustc_span::Span;

impl<'tcx> Mirror<'tcx> for &'tcx hir::Expr<'tcx> {
    type Output = Expr<'tcx>;

    fn make_mirror(self, cx: &mut Cx<'_, 'tcx>) -> Expr<'tcx> {
        let temp_lifetime = cx.region_scope_tree.temporary_scope(self.hir_id.local_id);
        let expr_scope = region::Scope { id: self.hir_id.local_id, data: region::ScopeData::Node };

        debug!("Expr::make_mirror(): id={}, span={:?}", self.hir_id, self.span);

        let mut expr = make_mirror_unadjusted(cx, self);

        // Now apply adjustments, if any.
        for adjustment in cx.typeck_results().expr_adjustments(self) {
            debug!("make_mirror: expr={:?} applying adjustment={:?}", expr, adjustment);
            expr = apply_adjustment(cx, self, expr, adjustment);
        }

        // Next, wrap this up in the expr's scope.
        expr = Expr {
            temp_lifetime,
            ty: expr.ty,
            span: self.span,
            kind: ExprKind::Scope {
                region_scope: expr_scope,
                value: expr.to_ref(),
                lint_level: LintLevel::Explicit(self.hir_id),
            },
        };

        // Finally, create a destruction scope, if any.
        if let Some(region_scope) = cx.region_scope_tree.opt_destruction_scope(self.hir_id.local_id)
        {
            expr = Expr {
                temp_lifetime,
                ty: expr.ty,
                span: self.span,
                kind: ExprKind::Scope {
                    region_scope,
                    value: expr.to_ref(),
                    lint_level: LintLevel::Inherited,
                },
            };
        }

        // OK, all done!
        expr
    }
}

fn apply_adjustment<'a, 'tcx>(
    cx: &mut Cx<'a, 'tcx>,
    hir_expr: &'tcx hir::Expr<'tcx>,
    mut expr: Expr<'tcx>,
    adjustment: &Adjustment<'tcx>,
) -> Expr<'tcx> {
    let Expr { temp_lifetime, mut span, .. } = expr;

    // Adjust the span from the block, to the last expression of the
    // block. This is a better span when returning a mutable reference
    // with too short a lifetime. The error message will use the span
    // from the assignment to the return place, which should only point
    // at the returned value, not the entire function body.
    //
    // fn return_short_lived<'a>(x: &'a mut i32) -> &'static mut i32 {
    //      x
    //   // ^ error message points at this expression.
    // }
    let mut adjust_span = |expr: &mut Expr<'tcx>| {
        if let ExprKind::Block { body } = expr.kind {
            if let Some(ref last_expr) = body.expr {
                span = last_expr.span;
                expr.span = span;
            }
        }
    };

    let kind = match adjustment.kind {
        Adjust::Pointer(PointerCast::Unsize) => {
            adjust_span(&mut expr);
            ExprKind::Pointer { cast: PointerCast::Unsize, source: expr.to_ref() }
        }
        Adjust::Pointer(cast) => ExprKind::Pointer { cast, source: expr.to_ref() },
        Adjust::NeverToAny => ExprKind::NeverToAny { source: expr.to_ref() },
        Adjust::Deref(None) => {
            adjust_span(&mut expr);
            ExprKind::Deref { arg: expr.to_ref() }
        }
        Adjust::Deref(Some(deref)) => {
            // We don't need to do call adjust_span here since
            // deref coercions always start with a built-in deref.
            let call = deref.method_call(cx.tcx(), expr.ty);

            expr = Expr {
                temp_lifetime,
                ty: cx.tcx.mk_ref(deref.region, ty::TypeAndMut { ty: expr.ty, mutbl: deref.mutbl }),
                span,
                kind: ExprKind::Borrow {
                    borrow_kind: deref.mutbl.to_borrow_kind(),
                    arg: expr.to_ref(),
                },
            };

            overloaded_place(
                cx,
                hir_expr,
                adjustment.target,
                Some(call),
                vec![expr.to_ref()],
                deref.span,
            )
        }
        Adjust::Borrow(AutoBorrow::Ref(_, m)) => {
            ExprKind::Borrow { borrow_kind: m.to_borrow_kind(), arg: expr.to_ref() }
        }
        Adjust::Borrow(AutoBorrow::RawPtr(mutability)) => {
            ExprKind::AddressOf { mutability, arg: expr.to_ref() }
        }
    };

    Expr { temp_lifetime, ty: adjustment.target, span, kind }
}

fn make_mirror_unadjusted<'a, 'tcx>(
    cx: &mut Cx<'a, 'tcx>,
    expr: &'tcx hir::Expr<'tcx>,
) -> Expr<'tcx> {
    let expr_ty = cx.typeck_results().expr_ty(expr);
    let temp_lifetime = cx.region_scope_tree.temporary_scope(expr.hir_id.local_id);

    let kind = match expr.kind {
        // Here comes the interesting stuff:
        hir::ExprKind::MethodCall(_, method_span, ref args, fn_span) => {
            // Rewrite a.b(c) into UFCS form like Trait::b(a, c)
            let expr = method_callee(cx, expr, method_span, None);
            let args = args.iter().map(|e| e.to_ref()).collect();
            ExprKind::Call { ty: expr.ty, fun: expr.to_ref(), args, from_hir_call: true, fn_span }
        }

        hir::ExprKind::Call(ref fun, ref args) => {
            if cx.typeck_results().is_method_call(expr) {
                // The callee is something implementing Fn, FnMut, or FnOnce.
                // Find the actual method implementation being called and
                // build the appropriate UFCS call expression with the
                // callee-object as expr parameter.

                // rewrite f(u, v) into FnOnce::call_once(f, (u, v))

                let method = method_callee(cx, expr, fun.span, None);

                let arg_tys = args.iter().map(|e| cx.typeck_results().expr_ty_adjusted(e));
                let tupled_args = Expr {
                    ty: cx.tcx.mk_tup(arg_tys),
                    temp_lifetime,
                    span: expr.span,
                    kind: ExprKind::Tuple { fields: args.iter().map(ToRef::to_ref).collect() },
                };

                ExprKind::Call {
                    ty: method.ty,
                    fun: method.to_ref(),
                    args: vec![fun.to_ref(), tupled_args.to_ref()],
                    from_hir_call: true,
                    fn_span: expr.span,
                }
            } else {
                let adt_data =
                    if let hir::ExprKind::Path(hir::QPath::Resolved(_, ref path)) = fun.kind {
                        // Tuple-like ADTs are represented as ExprKind::Call. We convert them here.
                        expr_ty.ty_adt_def().and_then(|adt_def| match path.res {
                            Res::Def(DefKind::Ctor(_, CtorKind::Fn), ctor_id) => {
                                Some((adt_def, adt_def.variant_index_with_ctor_id(ctor_id)))
                            }
                            Res::SelfCtor(..) => Some((adt_def, VariantIdx::new(0))),
                            _ => None,
                        })
                    } else {
                        None
                    };
                if let Some((adt_def, index)) = adt_data {
                    let substs = cx.typeck_results().node_substs(fun.hir_id);
                    let user_provided_types = cx.typeck_results().user_provided_types();
                    let user_ty = user_provided_types.get(fun.hir_id).copied().map(|mut u_ty| {
                        if let UserType::TypeOf(ref mut did, _) = &mut u_ty.value {
                            *did = adt_def.did;
                        }
                        u_ty
                    });
                    debug!("make_mirror_unadjusted: (call) user_ty={:?}", user_ty);

                    let field_refs = args
                        .iter()
                        .enumerate()
                        .map(|(idx, e)| FieldExprRef { name: Field::new(idx), expr: e.to_ref() })
                        .collect();
                    ExprKind::Adt {
                        adt_def,
                        substs,
                        variant_index: index,
                        fields: field_refs,
                        user_ty,
                        base: None,
                    }
                } else {
                    ExprKind::Call {
                        ty: cx.typeck_results().node_type(fun.hir_id),
                        fun: fun.to_ref(),
                        args: args.to_ref(),
                        from_hir_call: true,
                        fn_span: expr.span,
                    }
                }
            }
        }

        hir::ExprKind::AddrOf(hir::BorrowKind::Ref, mutbl, ref arg) => {
            ExprKind::Borrow { borrow_kind: mutbl.to_borrow_kind(), arg: arg.to_ref() }
        }

        hir::ExprKind::AddrOf(hir::BorrowKind::Raw, mutability, ref arg) => {
            ExprKind::AddressOf { mutability, arg: arg.to_ref() }
        }

        hir::ExprKind::Block(ref blk, _) => ExprKind::Block { body: &blk },

        hir::ExprKind::Assign(ref lhs, ref rhs, _) => {
            ExprKind::Assign { lhs: lhs.to_ref(), rhs: rhs.to_ref() }
        }

        hir::ExprKind::AssignOp(op, ref lhs, ref rhs) => {
            if cx.typeck_results().is_method_call(expr) {
                overloaded_operator(cx, expr, vec![lhs.to_ref(), rhs.to_ref()])
            } else {
                ExprKind::AssignOp { op: bin_op(op.node), lhs: lhs.to_ref(), rhs: rhs.to_ref() }
            }
        }

        hir::ExprKind::Lit(ref lit) => ExprKind::Literal {
            literal: cx.const_eval_literal(&lit.node, expr_ty, lit.span, false),
            user_ty: None,
            const_id: None,
        },

        hir::ExprKind::Binary(op, ref lhs, ref rhs) => {
            if cx.typeck_results().is_method_call(expr) {
                overloaded_operator(cx, expr, vec![lhs.to_ref(), rhs.to_ref()])
            } else {
                // FIXME overflow
                match (op.node, cx.constness) {
                    (hir::BinOpKind::And, _) => ExprKind::LogicalOp {
                        op: LogicalOp::And,
                        lhs: lhs.to_ref(),
                        rhs: rhs.to_ref(),
                    },
                    (hir::BinOpKind::Or, _) => ExprKind::LogicalOp {
                        op: LogicalOp::Or,
                        lhs: lhs.to_ref(),
                        rhs: rhs.to_ref(),
                    },

                    _ => {
                        let op = bin_op(op.node);
                        ExprKind::Binary { op, lhs: lhs.to_ref(), rhs: rhs.to_ref() }
                    }
                }
            }
        }

        hir::ExprKind::Index(ref lhs, ref index) => {
            if cx.typeck_results().is_method_call(expr) {
                overloaded_place(
                    cx,
                    expr,
                    expr_ty,
                    None,
                    vec![lhs.to_ref(), index.to_ref()],
                    expr.span,
                )
            } else {
                ExprKind::Index { lhs: lhs.to_ref(), index: index.to_ref() }
            }
        }

        hir::ExprKind::Unary(hir::UnOp::UnDeref, ref arg) => {
            if cx.typeck_results().is_method_call(expr) {
                overloaded_place(cx, expr, expr_ty, None, vec![arg.to_ref()], expr.span)
            } else {
                ExprKind::Deref { arg: arg.to_ref() }
            }
        }

        hir::ExprKind::Unary(hir::UnOp::UnNot, ref arg) => {
            if cx.typeck_results().is_method_call(expr) {
                overloaded_operator(cx, expr, vec![arg.to_ref()])
            } else {
                ExprKind::Unary { op: UnOp::Not, arg: arg.to_ref() }
            }
        }

        hir::ExprKind::Unary(hir::UnOp::UnNeg, ref arg) => {
            if cx.typeck_results().is_method_call(expr) {
                overloaded_operator(cx, expr, vec![arg.to_ref()])
            } else if let hir::ExprKind::Lit(ref lit) = arg.kind {
                ExprKind::Literal {
                    literal: cx.const_eval_literal(&lit.node, expr_ty, lit.span, true),
                    user_ty: None,
                    const_id: None,
                }
            } else {
                ExprKind::Unary { op: UnOp::Neg, arg: arg.to_ref() }
            }
        }

        hir::ExprKind::Struct(ref qpath, ref fields, ref base) => match expr_ty.kind() {
            ty::Adt(adt, substs) => match adt.adt_kind() {
                AdtKind::Struct | AdtKind::Union => {
                    let user_provided_types = cx.typeck_results().user_provided_types();
                    let user_ty = user_provided_types.get(expr.hir_id).copied();
                    debug!("make_mirror_unadjusted: (struct/union) user_ty={:?}", user_ty);
                    ExprKind::Adt {
                        adt_def: adt,
                        variant_index: VariantIdx::new(0),
                        substs,
                        user_ty,
                        fields: field_refs(cx, fields),
                        base: base.as_ref().map(|base| FruInfo {
                            base: base.to_ref(),
                            field_types: cx.typeck_results().fru_field_types()[expr.hir_id].clone(),
                        }),
                    }
                }
                AdtKind::Enum => {
                    let res = cx.typeck_results().qpath_res(qpath, expr.hir_id);
                    match res {
                        Res::Def(DefKind::Variant, variant_id) => {
                            assert!(base.is_none());

                            let index = adt.variant_index_with_id(variant_id);
                            let user_provided_types = cx.typeck_results().user_provided_types();
                            let user_ty = user_provided_types.get(expr.hir_id).copied();
                            debug!("make_mirror_unadjusted: (variant) user_ty={:?}", user_ty);
                            ExprKind::Adt {
                                adt_def: adt,
                                variant_index: index,
                                substs,
                                user_ty,
                                fields: field_refs(cx, fields),
                                base: None,
                            }
                        }
                        _ => {
                            span_bug!(expr.span, "unexpected res: {:?}", res);
                        }
                    }
                }
            },
            _ => {
                span_bug!(expr.span, "unexpected type for struct literal: {:?}", expr_ty);
            }
        },

        hir::ExprKind::Closure(..) => {
            let closure_ty = cx.typeck_results().expr_ty(expr);
            let (def_id, substs, movability) = match *closure_ty.kind() {
                ty::Closure(def_id, substs) => (def_id, UpvarSubsts::Closure(substs), None),
                ty::Generator(def_id, substs, movability) => {
                    (def_id, UpvarSubsts::Generator(substs), Some(movability))
                }
                _ => {
                    span_bug!(expr.span, "closure expr w/o closure type: {:?}", closure_ty);
                }
            };

            let upvars = cx
                .typeck_results()
                .closure_min_captures_flattened(def_id)
                .zip(substs.upvar_tys())
                .map(|(captured_place, ty)| capture_upvar(cx, expr, captured_place, ty))
                .collect();
            ExprKind::Closure { closure_id: def_id, substs, upvars, movability }
        }

        hir::ExprKind::Path(ref qpath) => {
            let res = cx.typeck_results().qpath_res(qpath, expr.hir_id);
            convert_path_expr(cx, expr, res)
        }

        hir::ExprKind::InlineAsm(ref asm) => ExprKind::InlineAsm {
            template: asm.template,
            operands: asm
                .operands
                .iter()
                .map(|(op, _op_sp)| {
                    match *op {
                        hir::InlineAsmOperand::In { reg, ref expr } => {
                            InlineAsmOperand::In { reg, expr: expr.to_ref() }
                        }
                        hir::InlineAsmOperand::Out { reg, late, ref expr } => {
                            InlineAsmOperand::Out {
                                reg,
                                late,
                                expr: expr.as_ref().map(|expr| expr.to_ref()),
                            }
                        }
                        hir::InlineAsmOperand::InOut { reg, late, ref expr } => {
                            InlineAsmOperand::InOut { reg, late, expr: expr.to_ref() }
                        }
                        hir::InlineAsmOperand::SplitInOut {
                            reg,
                            late,
                            ref in_expr,
                            ref out_expr,
                        } => InlineAsmOperand::SplitInOut {
                            reg,
                            late,
                            in_expr: in_expr.to_ref(),
                            out_expr: out_expr.as_ref().map(|expr| expr.to_ref()),
                        },
                        hir::InlineAsmOperand::Const { ref expr } => {
                            InlineAsmOperand::Const { expr: expr.to_ref() }
                        }
                        hir::InlineAsmOperand::Sym { ref expr } => {
                            let qpath = match expr.kind {
                                hir::ExprKind::Path(ref qpath) => qpath,
                                _ => span_bug!(
                                    expr.span,
                                    "asm `sym` operand should be a path, found {:?}",
                                    expr.kind
                                ),
                            };
                            let temp_lifetime =
                                cx.region_scope_tree.temporary_scope(expr.hir_id.local_id);
                            let res = cx.typeck_results().qpath_res(qpath, expr.hir_id);
                            let ty;
                            match res {
                                Res::Def(DefKind::Fn, _) | Res::Def(DefKind::AssocFn, _) => {
                                    ty = cx.typeck_results().node_type(expr.hir_id);
                                    let user_ty = user_substs_applied_to_res(cx, expr.hir_id, res);
                                    InlineAsmOperand::SymFn {
                                        expr: Expr {
                                            ty,
                                            temp_lifetime,
                                            span: expr.span,
                                            kind: ExprKind::Literal {
                                                literal: ty::Const::zero_sized(cx.tcx, ty),
                                                user_ty,
                                                const_id: None,
                                            },
                                        }
                                        .to_ref(),
                                    }
                                }

                                Res::Def(DefKind::Static, def_id) => {
                                    InlineAsmOperand::SymStatic { def_id }
                                }

                                _ => {
                                    cx.tcx.sess.span_err(
                                        expr.span,
                                        "asm `sym` operand must point to a fn or static",
                                    );

                                    // Not a real fn, but we're not reaching codegen anyways...
                                    ty = cx.tcx.ty_error();
                                    InlineAsmOperand::SymFn {
                                        expr: Expr {
                                            ty,
                                            temp_lifetime,
                                            span: expr.span,
                                            kind: ExprKind::Literal {
                                                literal: ty::Const::zero_sized(cx.tcx, ty),
                                                user_ty: None,
                                                const_id: None,
                                            },
                                        }
                                        .to_ref(),
                                    }
                                }
                            }
                        }
                    }
                })
                .collect(),
            options: asm.options,
            line_spans: asm.line_spans,
        },

        hir::ExprKind::LlvmInlineAsm(ref asm) => ExprKind::LlvmInlineAsm {
            asm: &asm.inner,
            outputs: asm.outputs_exprs.to_ref(),
            inputs: asm.inputs_exprs.to_ref(),
        },

        hir::ExprKind::ConstBlock(ref anon_const) => {
            let anon_const_def_id = cx.tcx.hir().local_def_id(anon_const.hir_id);
            let value = ty::Const::from_anon_const(cx.tcx, anon_const_def_id);

            ExprKind::ConstBlock { value }
        }
        // Now comes the rote stuff:
        hir::ExprKind::Repeat(ref v, ref count) => {
            let count_def_id = cx.tcx.hir().local_def_id(count.hir_id);
            let count = ty::Const::from_anon_const(cx.tcx, count_def_id);

            ExprKind::Repeat { value: v.to_ref(), count }
        }
        hir::ExprKind::Ret(ref v) => ExprKind::Return { value: v.to_ref() },
        hir::ExprKind::Break(dest, ref value) => match dest.target_id {
            Ok(target_id) => ExprKind::Break {
                label: region::Scope { id: target_id.local_id, data: region::ScopeData::Node },
                value: value.to_ref(),
            },
            Err(err) => bug!("invalid loop id for break: {}", err),
        },
        hir::ExprKind::Continue(dest) => match dest.target_id {
            Ok(loop_id) => ExprKind::Continue {
                label: region::Scope { id: loop_id.local_id, data: region::ScopeData::Node },
            },
            Err(err) => bug!("invalid loop id for continue: {}", err),
        },
        hir::ExprKind::Match(ref discr, ref arms, _) => ExprKind::Match {
            scrutinee: discr.to_ref(),
            arms: arms.iter().map(|a| convert_arm(cx, a)).collect(),
        },
        hir::ExprKind::Loop(ref body, _, _) => {
            ExprKind::Loop { body: block::to_expr_ref(cx, body) }
        }
        hir::ExprKind::Field(ref source, ..) => ExprKind::Field {
            lhs: source.to_ref(),
            name: Field::new(cx.tcx.field_index(expr.hir_id, cx.typeck_results)),
        },
        hir::ExprKind::Cast(ref source, ref cast_ty) => {
            // Check for a user-given type annotation on this `cast`
            let user_provided_types = cx.typeck_results.user_provided_types();
            let user_ty = user_provided_types.get(cast_ty.hir_id);

            debug!(
                "cast({:?}) has ty w/ hir_id {:?} and user provided ty {:?}",
                expr, cast_ty.hir_id, user_ty,
            );

            // Check to see if this cast is a "coercion cast", where the cast is actually done
            // using a coercion (or is a no-op).
            let cast = if cx.typeck_results().is_coercion_cast(source.hir_id) {
                // Convert the lexpr to a vexpr.
                ExprKind::Use { source: source.to_ref() }
            } else if cx.typeck_results().expr_ty(source).is_region_ptr() {
                // Special cased so that we can type check that the element
                // type of the source matches the pointed to type of the
                // destination.
                ExprKind::Pointer { source: source.to_ref(), cast: PointerCast::ArrayToPointer }
            } else {
                // check whether this is casting an enum variant discriminant
                // to prevent cycles, we refer to the discriminant initializer
                // which is always an integer and thus doesn't need to know the
                // enum's layout (or its tag type) to compute it during const eval
                // Example:
                // enum Foo {
                //     A,
                //     B = A as isize + 4,
                // }
                // The correct solution would be to add symbolic computations to miri,
                // so we wouldn't have to compute and store the actual value
                let var = if let hir::ExprKind::Path(ref qpath) = source.kind {
                    let res = cx.typeck_results().qpath_res(qpath, source.hir_id);
                    cx.typeck_results().node_type(source.hir_id).ty_adt_def().and_then(|adt_def| {
                        match res {
                            Res::Def(
                                DefKind::Ctor(CtorOf::Variant, CtorKind::Const),
                                variant_ctor_id,
                            ) => {
                                let idx = adt_def.variant_index_with_ctor_id(variant_ctor_id);
                                let (d, o) = adt_def.discriminant_def_for_variant(idx);
                                use rustc_middle::ty::util::IntTypeExt;
                                let ty = adt_def.repr.discr_type();
                                let ty = ty.to_ty(cx.tcx());
                                Some((d, o, ty))
                            }
                            _ => None,
                        }
                    })
                } else {
                    None
                };

                let source = if let Some((did, offset, var_ty)) = var {
                    let mk_const = |literal| {
                        Expr {
                            temp_lifetime,
                            ty: var_ty,
                            span: expr.span,
                            kind: ExprKind::Literal { literal, user_ty: None, const_id: None },
                        }
                        .to_ref()
                    };
                    let offset = mk_const(ty::Const::from_bits(
                        cx.tcx,
                        offset as u128,
                        cx.param_env.and(var_ty),
                    ));
                    match did {
                        Some(did) => {
                            // in case we are offsetting from a computed discriminant
                            // and not the beginning of discriminants (which is always `0`)
                            let substs = InternalSubsts::identity_for_item(cx.tcx(), did);
                            let lhs = mk_const(cx.tcx().mk_const(ty::Const {
                                val: ty::ConstKind::Unevaluated(
                                    ty::WithOptConstParam::unknown(did),
                                    substs,
                                    None,
                                ),
                                ty: var_ty,
                            }));
                            let bin = ExprKind::Binary { op: BinOp::Add, lhs, rhs: offset };
                            Expr { temp_lifetime, ty: var_ty, span: expr.span, kind: bin }.to_ref()
                        }
                        None => offset,
                    }
                } else {
                    source.to_ref()
                };

                ExprKind::Cast { source }
            };

            if let Some(user_ty) = user_ty {
                // NOTE: Creating a new Expr and wrapping a Cast inside of it may be
                //       inefficient, revisit this when performance becomes an issue.
                let cast_expr = Expr { temp_lifetime, ty: expr_ty, span: expr.span, kind: cast };
                debug!("make_mirror_unadjusted: (cast) user_ty={:?}", user_ty);

                ExprKind::ValueTypeAscription {
                    source: cast_expr.to_ref(),
                    user_ty: Some(*user_ty),
                }
            } else {
                cast
            }
        }
        hir::ExprKind::Type(ref source, ref ty) => {
            let user_provided_types = cx.typeck_results.user_provided_types();
            let user_ty = user_provided_types.get(ty.hir_id).copied();
            debug!("make_mirror_unadjusted: (type) user_ty={:?}", user_ty);
            if source.is_syntactic_place_expr() {
                ExprKind::PlaceTypeAscription { source: source.to_ref(), user_ty }
            } else {
                ExprKind::ValueTypeAscription { source: source.to_ref(), user_ty }
            }
        }
        hir::ExprKind::DropTemps(ref source) => ExprKind::Use { source: source.to_ref() },
        hir::ExprKind::Box(ref value) => ExprKind::Box { value: value.to_ref() },
        hir::ExprKind::Array(ref fields) => ExprKind::Array { fields: fields.to_ref() },
        hir::ExprKind::Tup(ref fields) => ExprKind::Tuple { fields: fields.to_ref() },

        hir::ExprKind::Yield(ref v, _) => ExprKind::Yield { value: v.to_ref() },
        hir::ExprKind::Err => unreachable!(),
    };

    Expr { temp_lifetime, ty: expr_ty, span: expr.span, kind }
}

fn user_substs_applied_to_res<'tcx>(
    cx: &mut Cx<'_, 'tcx>,
    hir_id: hir::HirId,
    res: Res,
) -> Option<ty::CanonicalUserType<'tcx>> {
    debug!("user_substs_applied_to_res: res={:?}", res);
    let user_provided_type = match res {
        // A reference to something callable -- e.g., a fn, method, or
        // a tuple-struct or tuple-variant. This has the type of a
        // `Fn` but with the user-given substitutions.
        Res::Def(DefKind::Fn, _)
        | Res::Def(DefKind::AssocFn, _)
        | Res::Def(DefKind::Ctor(_, CtorKind::Fn), _)
        | Res::Def(DefKind::Const, _)
        | Res::Def(DefKind::AssocConst, _) => {
            cx.typeck_results().user_provided_types().get(hir_id).copied()
        }

        // A unit struct/variant which is used as a value (e.g.,
        // `None`). This has the type of the enum/struct that defines
        // this variant -- but with the substitutions given by the
        // user.
        Res::Def(DefKind::Ctor(_, CtorKind::Const), _) => {
            cx.user_substs_applied_to_ty_of_hir_id(hir_id)
        }

        // `Self` is used in expression as a tuple struct constructor or an unit struct constructor
        Res::SelfCtor(_) => cx.user_substs_applied_to_ty_of_hir_id(hir_id),

        _ => bug!("user_substs_applied_to_res: unexpected res {:?} at {:?}", res, hir_id),
    };
    debug!("user_substs_applied_to_res: user_provided_type={:?}", user_provided_type);
    user_provided_type
}

fn method_callee<'a, 'tcx>(
    cx: &mut Cx<'a, 'tcx>,
    expr: &hir::Expr<'_>,
    span: Span,
    overloaded_callee: Option<(DefId, SubstsRef<'tcx>)>,
) -> Expr<'tcx> {
    let temp_lifetime = cx.region_scope_tree.temporary_scope(expr.hir_id.local_id);
    let (def_id, substs, user_ty) = match overloaded_callee {
        Some((def_id, substs)) => (def_id, substs, None),
        None => {
            let (kind, def_id) = cx
                .typeck_results()
                .type_dependent_def(expr.hir_id)
                .unwrap_or_else(|| span_bug!(expr.span, "no type-dependent def for method callee"));
            let user_ty = user_substs_applied_to_res(cx, expr.hir_id, Res::Def(kind, def_id));
            debug!("method_callee: user_ty={:?}", user_ty);
            (def_id, cx.typeck_results().node_substs(expr.hir_id), user_ty)
        }
    };
    let ty = cx.tcx().mk_fn_def(def_id, substs);
    Expr {
        temp_lifetime,
        ty,
        span,
        kind: ExprKind::Literal {
            literal: ty::Const::zero_sized(cx.tcx(), ty),
            user_ty,
            const_id: None,
        },
    }
}

trait ToBorrowKind {
    fn to_borrow_kind(&self) -> BorrowKind;
}

impl ToBorrowKind for AutoBorrowMutability {
    fn to_borrow_kind(&self) -> BorrowKind {
        use rustc_middle::ty::adjustment::AllowTwoPhase;
        match *self {
            AutoBorrowMutability::Mut { allow_two_phase_borrow } => BorrowKind::Mut {
                allow_two_phase_borrow: match allow_two_phase_borrow {
                    AllowTwoPhase::Yes => true,
                    AllowTwoPhase::No => false,
                },
            },
            AutoBorrowMutability::Not => BorrowKind::Shared,
        }
    }
}

impl ToBorrowKind for hir::Mutability {
    fn to_borrow_kind(&self) -> BorrowKind {
        match *self {
            hir::Mutability::Mut => BorrowKind::Mut { allow_two_phase_borrow: false },
            hir::Mutability::Not => BorrowKind::Shared,
        }
    }
}

fn convert_arm<'tcx>(cx: &mut Cx<'_, 'tcx>, arm: &'tcx hir::Arm<'tcx>) -> Arm<'tcx> {
    Arm {
        pattern: cx.pattern_from_hir(&arm.pat),
        guard: match arm.guard {
            Some(hir::Guard::If(ref e)) => Some(Guard::If(e.to_ref())),
            _ => None,
        },
        body: arm.body.to_ref(),
        lint_level: LintLevel::Explicit(arm.hir_id),
        scope: region::Scope { id: arm.hir_id.local_id, data: region::ScopeData::Node },
        span: arm.span,
    }
}

fn convert_path_expr<'a, 'tcx>(
    cx: &mut Cx<'a, 'tcx>,
    expr: &'tcx hir::Expr<'tcx>,
    res: Res,
) -> ExprKind<'tcx> {
    let substs = cx.typeck_results().node_substs(expr.hir_id);
    match res {
        // A regular function, constructor function or a constant.
        Res::Def(DefKind::Fn, _)
        | Res::Def(DefKind::AssocFn, _)
        | Res::Def(DefKind::Ctor(_, CtorKind::Fn), _)
        | Res::SelfCtor(..) => {
            let user_ty = user_substs_applied_to_res(cx, expr.hir_id, res);
            debug!("convert_path_expr: user_ty={:?}", user_ty);
            ExprKind::Literal {
                literal: ty::Const::zero_sized(cx.tcx, cx.typeck_results().node_type(expr.hir_id)),
                user_ty,
                const_id: None,
            }
        }

        Res::Def(DefKind::ConstParam, def_id) => {
            let hir_id = cx.tcx.hir().local_def_id_to_hir_id(def_id.expect_local());
            let item_id = cx.tcx.hir().get_parent_node(hir_id);
            let item_def_id = cx.tcx.hir().local_def_id(item_id);
            let generics = cx.tcx.generics_of(item_def_id);
            let local_def_id = cx.tcx.hir().local_def_id(hir_id);
            let index = generics.param_def_id_to_index[&local_def_id.to_def_id()];
            let name = cx.tcx.hir().name(hir_id);
            let val = ty::ConstKind::Param(ty::ParamConst::new(index, name));
            ExprKind::Literal {
                literal: cx
                    .tcx
                    .mk_const(ty::Const { val, ty: cx.typeck_results().node_type(expr.hir_id) }),
                user_ty: None,
                const_id: Some(def_id),
            }
        }

        Res::Def(DefKind::Const, def_id) | Res::Def(DefKind::AssocConst, def_id) => {
            let user_ty = user_substs_applied_to_res(cx, expr.hir_id, res);
            debug!("convert_path_expr: (const) user_ty={:?}", user_ty);
            ExprKind::Literal {
                literal: cx.tcx.mk_const(ty::Const {
                    val: ty::ConstKind::Unevaluated(
                        ty::WithOptConstParam::unknown(def_id),
                        substs,
                        None,
                    ),
                    ty: cx.typeck_results().node_type(expr.hir_id),
                }),
                user_ty,
                const_id: Some(def_id),
            }
        }

        Res::Def(DefKind::Ctor(_, CtorKind::Const), def_id) => {
            let user_provided_types = cx.typeck_results.user_provided_types();
            let user_provided_type = user_provided_types.get(expr.hir_id).copied();
            debug!("convert_path_expr: user_provided_type={:?}", user_provided_type);
            let ty = cx.typeck_results().node_type(expr.hir_id);
            match ty.kind() {
                // A unit struct/variant which is used as a value.
                // We return a completely different ExprKind here to account for this special case.
                ty::Adt(adt_def, substs) => ExprKind::Adt {
                    adt_def,
                    variant_index: adt_def.variant_index_with_ctor_id(def_id),
                    substs,
                    user_ty: user_provided_type,
                    fields: vec![],
                    base: None,
                },
                _ => bug!("unexpected ty: {:?}", ty),
            }
        }

        // We encode uses of statics as a `*&STATIC` where the `&STATIC` part is
        // a constant reference (or constant raw pointer for `static mut`) in MIR
        Res::Def(DefKind::Static, id) => {
            let ty = cx.tcx.static_ptr_ty(id);
            let temp_lifetime = cx.region_scope_tree.temporary_scope(expr.hir_id.local_id);
            let kind = if cx.tcx.is_thread_local_static(id) {
                ExprKind::ThreadLocalRef(id)
            } else {
                let ptr = cx.tcx.create_static_alloc(id);
                ExprKind::StaticRef {
                    literal: ty::Const::from_scalar(cx.tcx, Scalar::Ptr(ptr.into()), ty),
                    def_id: id,
                }
            };
            ExprKind::Deref { arg: Expr { ty, temp_lifetime, span: expr.span, kind }.to_ref() }
        }

        Res::Local(var_hir_id) => convert_var(cx, var_hir_id),

        _ => span_bug!(expr.span, "res `{:?}` not yet implemented", res),
    }
}

fn convert_var<'tcx>(cx: &mut Cx<'_, 'tcx>, var_hir_id: hir::HirId) -> ExprKind<'tcx> {
    // We want upvars here not captures.
    // Captures will be handled in MIR.
    let is_upvar = cx
        .tcx
        .upvars_mentioned(cx.body_owner)
        .map_or(false, |upvars| upvars.contains_key(&var_hir_id));

    debug!("convert_var({:?}): is_upvar={}, body_owner={:?}", var_hir_id, is_upvar, cx.body_owner);

    if is_upvar {
        ExprKind::UpvarRef { closure_def_id: cx.body_owner, var_hir_id }
    } else {
        ExprKind::VarRef { id: var_hir_id }
    }
}

fn bin_op(op: hir::BinOpKind) -> BinOp {
    match op {
        hir::BinOpKind::Add => BinOp::Add,
        hir::BinOpKind::Sub => BinOp::Sub,
        hir::BinOpKind::Mul => BinOp::Mul,
        hir::BinOpKind::Div => BinOp::Div,
        hir::BinOpKind::Rem => BinOp::Rem,
        hir::BinOpKind::BitXor => BinOp::BitXor,
        hir::BinOpKind::BitAnd => BinOp::BitAnd,
        hir::BinOpKind::BitOr => BinOp::BitOr,
        hir::BinOpKind::Shl => BinOp::Shl,
        hir::BinOpKind::Shr => BinOp::Shr,
        hir::BinOpKind::Eq => BinOp::Eq,
        hir::BinOpKind::Lt => BinOp::Lt,
        hir::BinOpKind::Le => BinOp::Le,
        hir::BinOpKind::Ne => BinOp::Ne,
        hir::BinOpKind::Ge => BinOp::Ge,
        hir::BinOpKind::Gt => BinOp::Gt,
        _ => bug!("no equivalent for ast binop {:?}", op),
    }
}

fn overloaded_operator<'a, 'tcx>(
    cx: &mut Cx<'a, 'tcx>,
    expr: &'tcx hir::Expr<'tcx>,
    args: Vec<ExprRef<'tcx>>,
) -> ExprKind<'tcx> {
    let fun = method_callee(cx, expr, expr.span, None);
    ExprKind::Call { ty: fun.ty, fun: fun.to_ref(), args, from_hir_call: false, fn_span: expr.span }
}

fn overloaded_place<'a, 'tcx>(
    cx: &mut Cx<'a, 'tcx>,
    expr: &'tcx hir::Expr<'tcx>,
    place_ty: Ty<'tcx>,
    overloaded_callee: Option<(DefId, SubstsRef<'tcx>)>,
    args: Vec<ExprRef<'tcx>>,
    span: Span,
) -> ExprKind<'tcx> {
    // For an overloaded *x or x[y] expression of type T, the method
    // call returns an &T and we must add the deref so that the types
    // line up (this is because `*x` and `x[y]` represent places):

    let recv_ty = match args[0] {
        ExprRef::Thir(e) => cx.typeck_results().expr_ty_adjusted(e),
        ExprRef::Mirror(ref e) => e.ty,
    };

    // Reconstruct the output assuming it's a reference with the
    // same region and mutability as the receiver. This holds for
    // `Deref(Mut)::Deref(_mut)` and `Index(Mut)::index(_mut)`.
    let (region, mutbl) = match *recv_ty.kind() {
        ty::Ref(region, _, mutbl) => (region, mutbl),
        _ => span_bug!(span, "overloaded_place: receiver is not a reference"),
    };
    let ref_ty = cx.tcx.mk_ref(region, ty::TypeAndMut { ty: place_ty, mutbl });

    // construct the complete expression `foo()` for the overloaded call,
    // which will yield the &T type
    let temp_lifetime = cx.region_scope_tree.temporary_scope(expr.hir_id.local_id);
    let fun = method_callee(cx, expr, span, overloaded_callee);
    let ref_expr = Expr {
        temp_lifetime,
        ty: ref_ty,
        span,
        kind: ExprKind::Call {
            ty: fun.ty,
            fun: fun.to_ref(),
            args,
            from_hir_call: false,
            fn_span: span,
        },
    };

    // construct and return a deref wrapper `*foo()`
    ExprKind::Deref { arg: ref_expr.to_ref() }
}

fn capture_upvar<'a, 'tcx>(
    cx: &mut Cx<'_, 'tcx>,
    closure_expr: &'tcx hir::Expr<'tcx>,
    captured_place: &'a ty::CapturedPlace<'tcx>,
    upvar_ty: Ty<'tcx>,
) -> ExprRef<'tcx> {
    let upvar_capture = captured_place.info.capture_kind;
    let temp_lifetime = cx.region_scope_tree.temporary_scope(closure_expr.hir_id.local_id);
    let var_ty = captured_place.place.base_ty;

    // The result of capture analysis in `rustc_typeck/check/upvar.rs`represents a captured path
    // as it's seen for use within the closure and not at the time of closure creation.
    //
    // That is we see expect to see it start from a captured upvar and not something that is local
    // to the closure's parent.
    let var_hir_id = match captured_place.place.base {
        HirPlaceBase::Upvar(upvar_id) => upvar_id.var_path.hir_id,
        base => bug!("Expected an upvar, found {:?}", base),
    };

    let mut captured_place_expr = Expr {
        temp_lifetime,
        ty: var_ty,
        span: closure_expr.span,
        kind: convert_var(cx, var_hir_id),
    };

    for proj in captured_place.place.projections.iter() {
        let kind = match proj.kind {
            HirProjectionKind::Deref => ExprKind::Deref { arg: captured_place_expr.to_ref() },
            HirProjectionKind::Field(field, ..) => {
                // Variant index will always be 0, because for multi-variant
                // enums, we capture the enum entirely.
                ExprKind::Field {
                    lhs: captured_place_expr.to_ref(),
                    name: Field::new(field as usize),
                }
            }
            HirProjectionKind::Index | HirProjectionKind::Subslice => {
                // We don't capture these projections, so we can ignore them here
                continue;
            }
        };

        captured_place_expr = Expr { temp_lifetime, ty: proj.ty, span: closure_expr.span, kind };
    }

    match upvar_capture {
        ty::UpvarCapture::ByValue(_) => captured_place_expr.to_ref(),
        ty::UpvarCapture::ByRef(upvar_borrow) => {
            let borrow_kind = match upvar_borrow.kind {
                ty::BorrowKind::ImmBorrow => BorrowKind::Shared,
                ty::BorrowKind::UniqueImmBorrow => BorrowKind::Unique,
                ty::BorrowKind::MutBorrow => BorrowKind::Mut { allow_two_phase_borrow: false },
            };
            Expr {
                temp_lifetime,
                ty: upvar_ty,
                span: closure_expr.span,
                kind: ExprKind::Borrow { borrow_kind, arg: captured_place_expr.to_ref() },
            }
            .to_ref()
        }
    }
}

/// Converts a list of named fields (i.e., for struct-like struct/enum ADTs) into FieldExprRef.
fn field_refs<'a, 'tcx>(
    cx: &mut Cx<'a, 'tcx>,
    fields: &'tcx [hir::Field<'tcx>],
) -> Vec<FieldExprRef<'tcx>> {
    fields
        .iter()
        .map(|field| FieldExprRef {
            name: Field::new(cx.tcx.field_index(field.hir_id, cx.typeck_results)),
            expr: field.expr.to_ref(),
        })
        .collect()
}
