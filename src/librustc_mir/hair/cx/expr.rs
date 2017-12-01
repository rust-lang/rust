// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use hair::*;
use rustc_data_structures::indexed_vec::Idx;
use rustc_const_math::ConstInt;
use hair::cx::Cx;
use hair::cx::block;
use hair::cx::to_ref::ToRef;
use rustc::hir::def::{Def, CtorKind};
use rustc::middle::const_val::ConstVal;
use rustc::ty::{self, AdtKind, VariantDef, Ty};
use rustc::ty::adjustment::{Adjustment, Adjust, AutoBorrow};
use rustc::ty::cast::CastKind as TyCastKind;
use rustc::hir;
use rustc::hir::def_id::LocalDefId;

impl<'tcx> Mirror<'tcx> for &'tcx hir::Expr {
    type Output = Expr<'tcx>;

    fn make_mirror<'a, 'gcx>(self, cx: &mut Cx<'a, 'gcx, 'tcx>) -> Expr<'tcx> {
        let temp_lifetime = cx.region_scope_tree.temporary_scope(self.hir_id.local_id);
        let expr_scope = region::Scope::Node(self.hir_id.local_id);

        debug!("Expr::make_mirror(): id={}, span={:?}", self.id, self.span);

        let mut expr = make_mirror_unadjusted(cx, self);

        // Now apply adjustments, if any.
        for adjustment in cx.tables().expr_adjustments(self) {
            debug!("make_mirror: expr={:?} applying adjustment={:?}",
                   expr,
                   adjustment);
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
                lint_level: cx.lint_level_of(self.id),
            },
        };

        // Finally, create a destruction scope, if any.
        if let Some(region_scope) =
            cx.region_scope_tree.opt_destruction_scope(self.hir_id.local_id) {
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

fn apply_adjustment<'a, 'gcx, 'tcx>(cx: &mut Cx<'a, 'gcx, 'tcx>,
                                    hir_expr: &'tcx hir::Expr,
                                    mut expr: Expr<'tcx>,
                                    adjustment: &Adjustment<'tcx>)
                                    -> Expr<'tcx> {
    let Expr { temp_lifetime, span, .. } = expr;
    let kind = match adjustment.kind {
        Adjust::ReifyFnPointer => {
            ExprKind::ReifyFnPointer { source: expr.to_ref() }
        }
        Adjust::UnsafeFnPointer => {
            ExprKind::UnsafeFnPointer { source: expr.to_ref() }
        }
        Adjust::ClosureFnPointer => {
            ExprKind::ClosureFnPointer { source: expr.to_ref() }
        }
        Adjust::NeverToAny => {
            ExprKind::NeverToAny { source: expr.to_ref() }
        }
        Adjust::MutToConstPointer => {
            ExprKind::Cast { source: expr.to_ref() }
        }
        Adjust::Deref(None) => {
            ExprKind::Deref { arg: expr.to_ref() }
        }
        Adjust::Deref(Some(deref)) => {
            let call = deref.method_call(cx.tcx, expr.ty);

            expr = Expr {
                temp_lifetime,
                ty: cx.tcx.mk_ref(deref.region,
                                  ty::TypeAndMut {
                                    ty: expr.ty,
                                    mutbl: deref.mutbl,
                                  }),
                span,
                kind: ExprKind::Borrow {
                    region: deref.region,
                    borrow_kind: to_borrow_kind(deref.mutbl),
                    arg: expr.to_ref(),
                },
            };

            overloaded_place(cx, hir_expr, adjustment.target, Some(call), vec![expr.to_ref()])
        }
        Adjust::Borrow(AutoBorrow::Ref(r, m)) => {
            ExprKind::Borrow {
                region: r,
                borrow_kind: to_borrow_kind(m),
                arg: expr.to_ref(),
            }
        }
        Adjust::Borrow(AutoBorrow::RawPtr(m)) => {
            // Convert this to a suitable `&foo` and
            // then an unsafe coercion. Limit the region to be just this
            // expression.
            let region = ty::ReScope(region::Scope::Node(hir_expr.hir_id.local_id));
            let region = cx.tcx.mk_region(region);
            expr = Expr {
                temp_lifetime,
                ty: cx.tcx.mk_ref(region,
                                  ty::TypeAndMut {
                                    ty: expr.ty,
                                    mutbl: m,
                                  }),
                span,
                kind: ExprKind::Borrow {
                    region,
                    borrow_kind: to_borrow_kind(m),
                    arg: expr.to_ref(),
                },
            };
            ExprKind::Cast { source: expr.to_ref() }
        }
        Adjust::Unsize => {
            ExprKind::Unsize { source: expr.to_ref() }
        }
    };

    Expr {
        temp_lifetime,
        ty: adjustment.target,
        span,
        kind,
    }
}

fn make_mirror_unadjusted<'a, 'gcx, 'tcx>(cx: &mut Cx<'a, 'gcx, 'tcx>,
                                          expr: &'tcx hir::Expr)
                                          -> Expr<'tcx> {
    let expr_ty = cx.tables().expr_ty(expr);
    let temp_lifetime = cx.region_scope_tree.temporary_scope(expr.hir_id.local_id);

    let kind = match expr.node {
        // Here comes the interesting stuff:
        hir::ExprMethodCall(.., ref args) => {
            // Rewrite a.b(c) into UFCS form like Trait::b(a, c)
            let expr = method_callee(cx, expr, None);
            let args = args.iter()
                .map(|e| e.to_ref())
                .collect();
            ExprKind::Call {
                ty: expr.ty,
                fun: expr.to_ref(),
                args,
            }
        }

        hir::ExprCall(ref fun, ref args) => {
            if cx.tables().is_method_call(expr) {
                // The callee is something implementing Fn, FnMut, or FnOnce.
                // Find the actual method implementation being called and
                // build the appropriate UFCS call expression with the
                // callee-object as expr parameter.

                // rewrite f(u, v) into FnOnce::call_once(f, (u, v))

                let method = method_callee(cx, expr, None);

                let arg_tys = args.iter().map(|e| cx.tables().expr_ty_adjusted(e));
                let tupled_args = Expr {
                    ty: cx.tcx.mk_tup(arg_tys, false),
                    temp_lifetime,
                    span: expr.span,
                    kind: ExprKind::Tuple { fields: args.iter().map(ToRef::to_ref).collect() },
                };

                ExprKind::Call {
                    ty: method.ty,
                    fun: method.to_ref(),
                    args: vec![fun.to_ref(), tupled_args.to_ref()],
                }
            } else {
                let adt_data = if let hir::ExprPath(hir::QPath::Resolved(_, ref path)) = fun.node {
                    // Tuple-like ADTs are represented as ExprCall. We convert them here.
                    expr_ty.ty_adt_def().and_then(|adt_def| {
                        match path.def {
                            Def::VariantCtor(variant_id, CtorKind::Fn) => {
                                Some((adt_def, adt_def.variant_index_with_id(variant_id)))
                            }
                            Def::StructCtor(_, CtorKind::Fn) => Some((adt_def, 0)),
                            _ => None,
                        }
                    })
                } else {
                    None
                };
                if let Some((adt_def, index)) = adt_data {
                    let substs = cx.tables().node_substs(fun.hir_id);
                    let field_refs = args.iter()
                        .enumerate()
                        .map(|(idx, e)| {
                            FieldExprRef {
                                name: Field::new(idx),
                                expr: e.to_ref(),
                            }
                        })
                        .collect();
                    ExprKind::Adt {
                        adt_def,
                        substs,
                        variant_index: index,
                        fields: field_refs,
                        base: None,
                    }
                } else {
                    ExprKind::Call {
                        ty: cx.tables().node_id_to_type(fun.hir_id),
                        fun: fun.to_ref(),
                        args: args.to_ref(),
                    }
                }
            }
        }

        hir::ExprAddrOf(mutbl, ref expr) => {
            let region = match expr_ty.sty {
                ty::TyRef(r, _) => r,
                _ => span_bug!(expr.span, "type of & not region"),
            };
            ExprKind::Borrow {
                region,
                borrow_kind: to_borrow_kind(mutbl),
                arg: expr.to_ref(),
            }
        }

        hir::ExprBlock(ref blk) => ExprKind::Block { body: &blk },

        hir::ExprAssign(ref lhs, ref rhs) => {
            ExprKind::Assign {
                lhs: lhs.to_ref(),
                rhs: rhs.to_ref(),
            }
        }

        hir::ExprAssignOp(op, ref lhs, ref rhs) => {
            if cx.tables().is_method_call(expr) {
                overloaded_operator(cx, expr, vec![lhs.to_ref(), rhs.to_ref()])
            } else {
                ExprKind::AssignOp {
                    op: bin_op(op.node),
                    lhs: lhs.to_ref(),
                    rhs: rhs.to_ref(),
                }
            }
        }

        hir::ExprLit(..) => ExprKind::Literal { literal: cx.const_eval_literal(expr) },

        hir::ExprBinary(op, ref lhs, ref rhs) => {
            if cx.tables().is_method_call(expr) {
                overloaded_operator(cx, expr, vec![lhs.to_ref(), rhs.to_ref()])
            } else {
                // FIXME overflow
                match (op.node, cx.constness) {
                    // FIXME(eddyb) use logical ops in constants when
                    // they can handle that kind of control-flow.
                    (hir::BinOp_::BiAnd, hir::Constness::Const) => {
                        ExprKind::Binary {
                            op: BinOp::BitAnd,
                            lhs: lhs.to_ref(),
                            rhs: rhs.to_ref(),
                        }
                    }
                    (hir::BinOp_::BiOr, hir::Constness::Const) => {
                        ExprKind::Binary {
                            op: BinOp::BitOr,
                            lhs: lhs.to_ref(),
                            rhs: rhs.to_ref(),
                        }
                    }

                    (hir::BinOp_::BiAnd, hir::Constness::NotConst) => {
                        ExprKind::LogicalOp {
                            op: LogicalOp::And,
                            lhs: lhs.to_ref(),
                            rhs: rhs.to_ref(),
                        }
                    }
                    (hir::BinOp_::BiOr, hir::Constness::NotConst) => {
                        ExprKind::LogicalOp {
                            op: LogicalOp::Or,
                            lhs: lhs.to_ref(),
                            rhs: rhs.to_ref(),
                        }
                    }

                    _ => {
                        let op = bin_op(op.node);
                        ExprKind::Binary {
                            op,
                            lhs: lhs.to_ref(),
                            rhs: rhs.to_ref(),
                        }
                    }
                }
            }
        }

        hir::ExprIndex(ref lhs, ref index) => {
            if cx.tables().is_method_call(expr) {
                overloaded_place(cx, expr, expr_ty, None, vec![lhs.to_ref(), index.to_ref()])
            } else {
                ExprKind::Index {
                    lhs: lhs.to_ref(),
                    index: index.to_ref(),
                }
            }
        }

        hir::ExprUnary(hir::UnOp::UnDeref, ref arg) => {
            if cx.tables().is_method_call(expr) {
                overloaded_place(cx, expr, expr_ty, None, vec![arg.to_ref()])
            } else {
                ExprKind::Deref { arg: arg.to_ref() }
            }
        }

        hir::ExprUnary(hir::UnOp::UnNot, ref arg) => {
            if cx.tables().is_method_call(expr) {
                overloaded_operator(cx, expr, vec![arg.to_ref()])
            } else {
                ExprKind::Unary {
                    op: UnOp::Not,
                    arg: arg.to_ref(),
                }
            }
        }

        hir::ExprUnary(hir::UnOp::UnNeg, ref arg) => {
            if cx.tables().is_method_call(expr) {
                overloaded_operator(cx, expr, vec![arg.to_ref()])
            } else {
                // FIXME runtime-overflow
                if let hir::ExprLit(_) = arg.node {
                    ExprKind::Literal { literal: cx.const_eval_literal(expr) }
                } else {
                    ExprKind::Unary {
                        op: UnOp::Neg,
                        arg: arg.to_ref(),
                    }
                }
            }
        }

        hir::ExprStruct(ref qpath, ref fields, ref base) => {
            match expr_ty.sty {
                ty::TyAdt(adt, substs) => {
                    match adt.adt_kind() {
                        AdtKind::Struct | AdtKind::Union => {
                            let field_refs = field_refs(&adt.variants[0], fields);
                            ExprKind::Adt {
                                adt_def: adt,
                                variant_index: 0,
                                substs,
                                fields: field_refs,
                                base: base.as_ref().map(|base| {
                                    FruInfo {
                                        base: base.to_ref(),
                                        field_types: cx.tables()
                                                       .fru_field_types()[expr.hir_id]
                                                       .clone(),
                                    }
                                }),
                            }
                        }
                        AdtKind::Enum => {
                            let def = match *qpath {
                                hir::QPath::Resolved(_, ref path) => path.def,
                                hir::QPath::TypeRelative(..) => Def::Err,
                            };
                            match def {
                                Def::Variant(variant_id) => {
                                    assert!(base.is_none());

                                    let index = adt.variant_index_with_id(variant_id);
                                    let field_refs = field_refs(&adt.variants[index], fields);
                                    ExprKind::Adt {
                                        adt_def: adt,
                                        variant_index: index,
                                        substs,
                                        fields: field_refs,
                                        base: None,
                                    }
                                }
                                _ => {
                                    span_bug!(expr.span, "unexpected def: {:?}", def);
                                }
                            }
                        }
                    }
                }
                _ => {
                    span_bug!(expr.span,
                              "unexpected type for struct literal: {:?}",
                              expr_ty);
                }
            }
        }

        hir::ExprClosure(..) => {
            let closure_ty = cx.tables().expr_ty(expr);
            let (def_id, substs, interior) = match closure_ty.sty {
                ty::TyClosure(def_id, substs) => (def_id, substs, None),
                ty::TyGenerator(def_id, substs, interior) => (def_id, substs, Some(interior)),
                _ => {
                    span_bug!(expr.span, "closure expr w/o closure type: {:?}", closure_ty);
                }
            };
            let upvars = cx.tcx.with_freevars(expr.id, |freevars| {
                freevars.iter()
                    .zip(substs.upvar_tys(def_id, cx.tcx))
                    .map(|(fv, ty)| capture_freevar(cx, expr, fv, ty))
                    .collect()
            });
            ExprKind::Closure {
                closure_id: def_id,
                substs,
                upvars,
                interior,
            }
        }

        hir::ExprPath(ref qpath) => {
            let def = cx.tables().qpath_def(qpath, expr.hir_id);
            convert_path_expr(cx, expr, def)
        }

        hir::ExprInlineAsm(ref asm, ref outputs, ref inputs) => {
            ExprKind::InlineAsm {
                asm,
                outputs: outputs.to_ref(),
                inputs: inputs.to_ref(),
            }
        }

        // Now comes the rote stuff:
        hir::ExprRepeat(ref v, count) => {
            let c = &cx.tcx.hir.body(count).value;
            let def_id = cx.tcx.hir.body_owner_def_id(count);
            let substs = Substs::identity_for_item(cx.tcx.global_tcx(), def_id);
            let count = match cx.tcx.at(c.span).const_eval(cx.param_env.and((def_id, substs))) {
                Ok(&ty::Const { val: ConstVal::Integral(ConstInt::Usize(u)), .. }) => u,
                Ok(other) => bug!("constant evaluation of repeat count yielded {:?}", other),
                Err(s) => cx.fatal_const_eval_err(&s, c.span, "expression")
            };

            ExprKind::Repeat {
                value: v.to_ref(),
                count,
            }
        }
        hir::ExprRet(ref v) => ExprKind::Return { value: v.to_ref() },
        hir::ExprBreak(dest, ref value) => {
            match dest.target_id {
                hir::ScopeTarget::Block(target_id) |
                hir::ScopeTarget::Loop(hir::LoopIdResult::Ok(target_id)) => ExprKind::Break {
                    label: region::Scope::Node(cx.tcx.hir.node_to_hir_id(target_id).local_id),
                    value: value.to_ref(),
                },
                hir::ScopeTarget::Loop(hir::LoopIdResult::Err(err)) =>
                    bug!("invalid loop id for break: {}", err)
            }
        }
        hir::ExprAgain(dest) => {
            match dest.target_id {
                hir::ScopeTarget::Block(_) => bug!("cannot continue to blocks"),
                hir::ScopeTarget::Loop(hir::LoopIdResult::Ok(loop_id)) => ExprKind::Continue {
                    label: region::Scope::Node(cx.tcx.hir.node_to_hir_id(loop_id).local_id),
                },
                hir::ScopeTarget::Loop(hir::LoopIdResult::Err(err)) =>
                    bug!("invalid loop id for continue: {}", err)
            }
        }
        hir::ExprMatch(ref discr, ref arms, _) => {
            ExprKind::Match {
                discriminant: discr.to_ref(),
                arms: arms.iter().map(|a| convert_arm(cx, a)).collect(),
            }
        }
        hir::ExprIf(ref cond, ref then, ref otherwise) => {
            ExprKind::If {
                condition: cond.to_ref(),
                then: then.to_ref(),
                otherwise: otherwise.to_ref(),
            }
        }
        hir::ExprWhile(ref cond, ref body, _) => {
            ExprKind::Loop {
                condition: Some(cond.to_ref()),
                body: block::to_expr_ref(cx, body),
            }
        }
        hir::ExprLoop(ref body, _, _) => {
            ExprKind::Loop {
                condition: None,
                body: block::to_expr_ref(cx, body),
            }
        }
        hir::ExprField(ref source, name) => {
            let index = match cx.tables().expr_ty_adjusted(source).sty {
                ty::TyAdt(adt_def, _) => adt_def.variants[0].index_of_field_named(name.node),
                ref ty => span_bug!(expr.span, "field of non-ADT: {:?}", ty),
            };
            let index =
                index.unwrap_or_else(|| {
                    span_bug!(expr.span, "no index found for field `{}`", name.node)
                });
            ExprKind::Field {
                lhs: source.to_ref(),
                name: Field::new(index),
            }
        }
        hir::ExprTupField(ref source, index) => {
            ExprKind::Field {
                lhs: source.to_ref(),
                name: Field::new(index.node as usize),
            }
        }
        hir::ExprCast(ref source, _) => {
            // Check to see if this cast is a "coercion cast", where the cast is actually done
            // using a coercion (or is a no-op).
            if let Some(&TyCastKind::CoercionCast) = cx.tables()
                                                       .cast_kinds()
                                                       .get(source.hir_id) {
                // Convert the lexpr to a vexpr.
                ExprKind::Use { source: source.to_ref() }
            } else {
                ExprKind::Cast { source: source.to_ref() }
            }
        }
        hir::ExprType(ref source, _) => return source.make_mirror(cx),
        hir::ExprBox(ref value) => {
            ExprKind::Box {
                value: value.to_ref(),
            }
        }
        hir::ExprArray(ref fields) => ExprKind::Array { fields: fields.to_ref() },
        hir::ExprTup(ref fields) => ExprKind::Tuple { fields: fields.to_ref() },

        hir::ExprYield(ref v) => ExprKind::Yield { value: v.to_ref() },
    };

    Expr {
        temp_lifetime,
        ty: expr_ty,
        span: expr.span,
        kind,
    }
}

fn method_callee<'a, 'gcx, 'tcx>(cx: &mut Cx<'a, 'gcx, 'tcx>,
                                 expr: &hir::Expr,
                                 custom_callee: Option<(DefId, &'tcx Substs<'tcx>)>)
                                 -> Expr<'tcx> {
    let temp_lifetime = cx.region_scope_tree.temporary_scope(expr.hir_id.local_id);
    let (def_id, substs) = custom_callee.unwrap_or_else(|| {
        (cx.tables().type_dependent_defs()[expr.hir_id].def_id(),
         cx.tables().node_substs(expr.hir_id))
    });
    let ty = cx.tcx().mk_fn_def(def_id, substs);
    Expr {
        temp_lifetime,
        ty,
        span: expr.span,
        kind: ExprKind::Literal {
            literal: Literal::Value {
                value: cx.tcx.mk_const(ty::Const {
                    val: ConstVal::Function(def_id, substs),
                    ty
                }),
            },
        },
    }
}

fn to_borrow_kind(m: hir::Mutability) -> BorrowKind {
    match m {
        hir::MutMutable => BorrowKind::Mut,
        hir::MutImmutable => BorrowKind::Shared,
    }
}

fn convert_arm<'a, 'gcx, 'tcx>(cx: &mut Cx<'a, 'gcx, 'tcx>, arm: &'tcx hir::Arm) -> Arm<'tcx> {
    Arm {
        patterns: arm.pats.iter().map(|p| cx.pattern_from_hir(p)).collect(),
        guard: arm.guard.to_ref(),
        body: arm.body.to_ref(),
        // BUG: fix this
        lint_level: LintLevel::Inherited,
    }
}

fn convert_path_expr<'a, 'gcx, 'tcx>(cx: &mut Cx<'a, 'gcx, 'tcx>,
                                     expr: &'tcx hir::Expr,
                                     def: Def)
                                     -> ExprKind<'tcx> {
    let substs = cx.tables().node_substs(expr.hir_id);
    match def {
        // A regular function, constructor function or a constant.
        Def::Fn(def_id) |
        Def::Method(def_id) |
        Def::StructCtor(def_id, CtorKind::Fn) |
        Def::VariantCtor(def_id, CtorKind::Fn) => ExprKind::Literal {
            literal: Literal::Value {
                value: cx.tcx.mk_const(ty::Const {
                    val: ConstVal::Function(def_id, substs),
                    ty: cx.tables().node_id_to_type(expr.hir_id)
                }),
            },
        },

        Def::Const(def_id) |
        Def::AssociatedConst(def_id) => ExprKind::Literal {
            literal: Literal::Value {
                value: cx.tcx.mk_const(ty::Const {
                    val: ConstVal::Unevaluated(def_id, substs),
                    ty: cx.tables().node_id_to_type(expr.hir_id)
                }),
            },
        },

        Def::StructCtor(def_id, CtorKind::Const) |
        Def::VariantCtor(def_id, CtorKind::Const) => {
            match cx.tables().node_id_to_type(expr.hir_id).sty {
                // A unit struct/variant which is used as a value.
                // We return a completely different ExprKind here to account for this special case.
                ty::TyAdt(adt_def, substs) => {
                    ExprKind::Adt {
                        adt_def,
                        variant_index: adt_def.variant_index_with_id(def_id),
                        substs,
                        fields: vec![],
                        base: None,
                    }
                }
                ref sty => bug!("unexpected sty: {:?}", sty),
            }
        }

        Def::Static(node_id, _) => ExprKind::StaticRef { id: node_id },

        Def::Local(..) | Def::Upvar(..) => convert_var(cx, expr, def),

        _ => span_bug!(expr.span, "def `{:?}` not yet implemented", def),
    }
}

fn convert_var<'a, 'gcx, 'tcx>(cx: &mut Cx<'a, 'gcx, 'tcx>,
                               expr: &'tcx hir::Expr,
                               def: Def)
                               -> ExprKind<'tcx> {
    let temp_lifetime = cx.region_scope_tree.temporary_scope(expr.hir_id.local_id);

    match def {
        Def::Local(id) => ExprKind::VarRef { id },

        Def::Upvar(var_id, index, closure_expr_id) => {
            debug!("convert_var(upvar({:?}, {:?}, {:?}))",
                   var_id,
                   index,
                   closure_expr_id);
            let var_hir_id = cx.tcx.hir.node_to_hir_id(var_id);
            let var_ty = cx.tables().node_id_to_type(var_hir_id);

            // FIXME free regions in closures are not right
            let closure_ty = cx.tables()
                               .node_id_to_type(cx.tcx.hir.node_to_hir_id(closure_expr_id));

            // FIXME we're just hard-coding the idea that the
            // signature will be &self or &mut self and hence will
            // have a bound region with number 0
            let closure_def_id = cx.tcx.hir.local_def_id(closure_expr_id);
            let region = ty::ReFree(ty::FreeRegion {
                scope: closure_def_id,
                bound_region: ty::BoundRegion::BrAnon(0),
            });
            let region = cx.tcx.mk_region(region);

            let self_expr = if let ty::TyClosure(_, closure_substs) = closure_ty.sty {
                match cx.infcx.closure_kind(closure_def_id, closure_substs).unwrap() {
                    ty::ClosureKind::Fn => {
                        let ref_closure_ty = cx.tcx.mk_ref(region,
                                                           ty::TypeAndMut {
                                                               ty: closure_ty,
                                                               mutbl: hir::MutImmutable,
                                                           });
                        Expr {
                            ty: closure_ty,
                            temp_lifetime: temp_lifetime,
                            span: expr.span,
                            kind: ExprKind::Deref {
                                arg: Expr {
                                    ty: ref_closure_ty,
                                    temp_lifetime,
                                    span: expr.span,
                                    kind: ExprKind::SelfRef,
                                }
                                .to_ref(),
                            },
                        }
                    }
                    ty::ClosureKind::FnMut => {
                        let ref_closure_ty = cx.tcx.mk_ref(region,
                                                           ty::TypeAndMut {
                                                               ty: closure_ty,
                                                               mutbl: hir::MutMutable,
                                                           });
                        Expr {
                            ty: closure_ty,
                            temp_lifetime,
                            span: expr.span,
                            kind: ExprKind::Deref {
                                arg: Expr {
                                    ty: ref_closure_ty,
                                    temp_lifetime,
                                    span: expr.span,
                                    kind: ExprKind::SelfRef,
                                }.to_ref(),
                            },
                        }
                    }
                    ty::ClosureKind::FnOnce => {
                        Expr {
                            ty: closure_ty,
                            temp_lifetime,
                            span: expr.span,
                            kind: ExprKind::SelfRef,
                        }
                    }
                }
            } else {
                Expr {
                    ty: closure_ty,
                    temp_lifetime,
                    span: expr.span,
                    kind: ExprKind::SelfRef,
                }
            };

            // at this point we have `self.n`, which loads up the upvar
            let field_kind = ExprKind::Field {
                lhs: self_expr.to_ref(),
                name: Field::new(index),
            };

            // ...but the upvar might be an `&T` or `&mut T` capture, at which
            // point we need an implicit deref
            let upvar_id = ty::UpvarId {
                var_id: var_hir_id,
                closure_expr_id: LocalDefId::from_def_id(closure_def_id),
            };
            match cx.tables().upvar_capture(upvar_id) {
                ty::UpvarCapture::ByValue => field_kind,
                ty::UpvarCapture::ByRef(borrow) => {
                    ExprKind::Deref {
                        arg: Expr {
                            temp_lifetime,
                            ty: cx.tcx.mk_ref(borrow.region,
                                              ty::TypeAndMut {
                                                  ty: var_ty,
                                                  mutbl: borrow.kind.to_mutbl_lossy(),
                                              }),
                            span: expr.span,
                            kind: field_kind,
                        }.to_ref(),
                    }
                }
            }
        }

        _ => span_bug!(expr.span, "type of & not region"),
    }
}


fn bin_op(op: hir::BinOp_) -> BinOp {
    match op {
        hir::BinOp_::BiAdd => BinOp::Add,
        hir::BinOp_::BiSub => BinOp::Sub,
        hir::BinOp_::BiMul => BinOp::Mul,
        hir::BinOp_::BiDiv => BinOp::Div,
        hir::BinOp_::BiRem => BinOp::Rem,
        hir::BinOp_::BiBitXor => BinOp::BitXor,
        hir::BinOp_::BiBitAnd => BinOp::BitAnd,
        hir::BinOp_::BiBitOr => BinOp::BitOr,
        hir::BinOp_::BiShl => BinOp::Shl,
        hir::BinOp_::BiShr => BinOp::Shr,
        hir::BinOp_::BiEq => BinOp::Eq,
        hir::BinOp_::BiLt => BinOp::Lt,
        hir::BinOp_::BiLe => BinOp::Le,
        hir::BinOp_::BiNe => BinOp::Ne,
        hir::BinOp_::BiGe => BinOp::Ge,
        hir::BinOp_::BiGt => BinOp::Gt,
        _ => bug!("no equivalent for ast binop {:?}", op),
    }
}

fn overloaded_operator<'a, 'gcx, 'tcx>(cx: &mut Cx<'a, 'gcx, 'tcx>,
                                       expr: &'tcx hir::Expr,
                                       args: Vec<ExprRef<'tcx>>)
                                       -> ExprKind<'tcx> {
    let fun = method_callee(cx, expr, None);
    ExprKind::Call {
        ty: fun.ty,
        fun: fun.to_ref(),
        args,
    }
}

fn overloaded_place<'a, 'gcx, 'tcx>(cx: &mut Cx<'a, 'gcx, 'tcx>,
                                     expr: &'tcx hir::Expr,
                                     place_ty: Ty<'tcx>,
                                     custom_callee: Option<(DefId, &'tcx Substs<'tcx>)>,
                                     args: Vec<ExprRef<'tcx>>)
                                     -> ExprKind<'tcx> {
    // For an overloaded *x or x[y] expression of type T, the method
    // call returns an &T and we must add the deref so that the types
    // line up (this is because `*x` and `x[y]` represent places):

    let recv_ty = match args[0] {
        ExprRef::Hair(e) => cx.tables().expr_ty_adjusted(e),
        ExprRef::Mirror(ref e) => e.ty
    };

    // Reconstruct the output assuming it's a reference with the
    // same region and mutability as the receiver. This holds for
    // `Deref(Mut)::Deref(_mut)` and `Index(Mut)::index(_mut)`.
    let (region, mt) = match recv_ty.sty {
        ty::TyRef(region, mt) => (region, mt),
        _ => span_bug!(expr.span, "overloaded_place: receiver is not a reference"),
    };
    let ref_ty = cx.tcx.mk_ref(region, ty::TypeAndMut {
        ty: place_ty,
        mutbl: mt.mutbl,
    });

    // construct the complete expression `foo()` for the overloaded call,
    // which will yield the &T type
    let temp_lifetime = cx.region_scope_tree.temporary_scope(expr.hir_id.local_id);
    let fun = method_callee(cx, expr, custom_callee);
    let ref_expr = Expr {
        temp_lifetime,
        ty: ref_ty,
        span: expr.span,
        kind: ExprKind::Call {
            ty: fun.ty,
            fun: fun.to_ref(),
            args,
        },
    };

    // construct and return a deref wrapper `*foo()`
    ExprKind::Deref { arg: ref_expr.to_ref() }
}

fn capture_freevar<'a, 'gcx, 'tcx>(cx: &mut Cx<'a, 'gcx, 'tcx>,
                                   closure_expr: &'tcx hir::Expr,
                                   freevar: &hir::Freevar,
                                   freevar_ty: Ty<'tcx>)
                                   -> ExprRef<'tcx> {
    let var_hir_id = cx.tcx.hir.node_to_hir_id(freevar.var_id());
    let upvar_id = ty::UpvarId {
        var_id: var_hir_id,
        closure_expr_id: cx.tcx.hir.local_def_id(closure_expr.id).to_local(),
    };
    let upvar_capture = cx.tables().upvar_capture(upvar_id);
    let temp_lifetime = cx.region_scope_tree.temporary_scope(closure_expr.hir_id.local_id);
    let var_ty = cx.tables().node_id_to_type(var_hir_id);
    let captured_var = Expr {
        temp_lifetime,
        ty: var_ty,
        span: closure_expr.span,
        kind: convert_var(cx, closure_expr, freevar.def),
    };
    match upvar_capture {
        ty::UpvarCapture::ByValue => captured_var.to_ref(),
        ty::UpvarCapture::ByRef(upvar_borrow) => {
            let borrow_kind = match upvar_borrow.kind {
                ty::BorrowKind::ImmBorrow => BorrowKind::Shared,
                ty::BorrowKind::UniqueImmBorrow => BorrowKind::Unique,
                ty::BorrowKind::MutBorrow => BorrowKind::Mut,
            };
            Expr {
                temp_lifetime,
                ty: freevar_ty,
                span: closure_expr.span,
                kind: ExprKind::Borrow {
                    region: upvar_borrow.region,
                    borrow_kind,
                    arg: captured_var.to_ref(),
                },
            }.to_ref()
        }
    }
}

/// Converts a list of named fields (i.e. for struct-like struct/enum ADTs) into FieldExprRef.
fn field_refs<'tcx>(variant: &'tcx VariantDef,
                    fields: &'tcx [hir::Field])
                    -> Vec<FieldExprRef<'tcx>> {
    fields.iter()
        .map(|field| {
            FieldExprRef {
                name: Field::new(variant.index_of_field_named(field.name.node).unwrap()),
                expr: field.expr.to_ref(),
            }
        })
        .collect()
}
