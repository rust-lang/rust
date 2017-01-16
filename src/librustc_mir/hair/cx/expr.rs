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
use rustc::hir::map;
use rustc::hir::def::{Def, CtorKind};
use rustc::middle::const_val::ConstVal;
use rustc_const_eval::{ConstContext, EvalHint, fatal_const_eval_err};
use rustc::ty::{self, AdtKind, VariantDef, Ty};
use rustc::ty::cast::CastKind as TyCastKind;
use rustc::hir;
use syntax::ptr::P;

impl<'tcx> Mirror<'tcx> for &'tcx hir::Expr {
    type Output = Expr<'tcx>;

    fn make_mirror<'a, 'gcx>(self, cx: &mut Cx<'a, 'gcx, 'tcx>) -> Expr<'tcx> {
        let temp_lifetime = cx.tcx.region_maps.temporary_scope(self.id);
        let expr_extent = cx.tcx.region_maps.node_extent(self.id);

        debug!("Expr::make_mirror(): id={}, span={:?}", self.id, self.span);

        let mut expr = make_mirror_unadjusted(cx, self);
        let adj = cx.tables().adjustments.get(&self.id).cloned();

        debug!("make_mirror: unadjusted-expr={:?} applying adjustments={:?}",
               expr,
               adj);

        // Now apply adjustments, if any.
        match adj.map(|adj| (adj.kind, adj.target)) {
            None => {}
            Some((ty::adjustment::Adjust::ReifyFnPointer, adjusted_ty)) => {
                expr = Expr {
                    temp_lifetime: temp_lifetime,
                    ty: adjusted_ty,
                    span: self.span,
                    kind: ExprKind::ReifyFnPointer { source: expr.to_ref() },
                };
            }
            Some((ty::adjustment::Adjust::UnsafeFnPointer, adjusted_ty)) => {
                expr = Expr {
                    temp_lifetime: temp_lifetime,
                    ty: adjusted_ty,
                    span: self.span,
                    kind: ExprKind::UnsafeFnPointer { source: expr.to_ref() },
                };
            }
            Some((ty::adjustment::Adjust::NeverToAny, adjusted_ty)) => {
                expr = Expr {
                    temp_lifetime: temp_lifetime,
                    ty: adjusted_ty,
                    span: self.span,
                    kind: ExprKind::NeverToAny { source: expr.to_ref() },
                };
            }
            Some((ty::adjustment::Adjust::MutToConstPointer, adjusted_ty)) => {
                expr = Expr {
                    temp_lifetime: temp_lifetime,
                    ty: adjusted_ty,
                    span: self.span,
                    kind: ExprKind::Cast { source: expr.to_ref() },
                };
            }
            Some((ty::adjustment::Adjust::DerefRef { autoderefs, autoref, unsize },
                  adjusted_ty)) => {
                for i in 0..autoderefs {
                    let i = i as u32;
                    let adjusted_ty =
                        expr.ty.adjust_for_autoderef(cx.tcx, self.id, self.span, i, |mc| {
                            cx.tables().method_map.get(&mc).map(|m| m.ty)
                        });
                    debug!("make_mirror: autoderef #{}, adjusted_ty={:?}",
                           i,
                           adjusted_ty);
                    let method_key = ty::MethodCall::autoderef(self.id, i);
                    let meth_ty = cx.tables().method_map.get(&method_key).map(|m| m.ty);
                    let kind = if let Some(meth_ty) = meth_ty {
                        debug!("make_mirror: overloaded autoderef (meth_ty={:?})", meth_ty);

                        let ref_ty = cx.tcx.no_late_bound_regions(&meth_ty.fn_ret());
                        let (region, mutbl) = match ref_ty {
                            Some(&ty::TyS { sty: ty::TyRef(region, mt), .. }) => (region, mt.mutbl),
                            _ => span_bug!(expr.span, "autoderef returned bad type"),
                        };

                        expr = Expr {
                            temp_lifetime: temp_lifetime,
                            ty: cx.tcx.mk_ref(region,
                                              ty::TypeAndMut {
                                                  ty: expr.ty,
                                                  mutbl: mutbl,
                                              }),
                            span: expr.span,
                            kind: ExprKind::Borrow {
                                region: region,
                                borrow_kind: to_borrow_kind(mutbl),
                                arg: expr.to_ref(),
                            },
                        };

                        overloaded_lvalue(cx,
                                          self,
                                          method_key,
                                          PassArgs::ByRef,
                                          expr.to_ref(),
                                          vec![])
                    } else {
                        debug!("make_mirror: built-in autoderef");
                        ExprKind::Deref { arg: expr.to_ref() }
                    };
                    expr = Expr {
                        temp_lifetime: temp_lifetime,
                        ty: adjusted_ty,
                        span: self.span,
                        kind: kind,
                    };
                }

                if let Some(autoref) = autoref {
                    let adjusted_ty = expr.ty.adjust_for_autoref(cx.tcx, Some(autoref));
                    match autoref {
                        ty::adjustment::AutoBorrow::Ref(r, m) => {
                            expr = Expr {
                                temp_lifetime: temp_lifetime,
                                ty: adjusted_ty,
                                span: self.span,
                                kind: ExprKind::Borrow {
                                    region: r,
                                    borrow_kind: to_borrow_kind(m),
                                    arg: expr.to_ref(),
                                },
                            };
                        }
                        ty::adjustment::AutoBorrow::RawPtr(m) => {
                            // Convert this to a suitable `&foo` and
                            // then an unsafe coercion. Limit the region to be just this
                            // expression.
                            let region = ty::ReScope(expr_extent);
                            let region = cx.tcx.mk_region(region);
                            expr = Expr {
                                temp_lifetime: temp_lifetime,
                                ty: cx.tcx.mk_ref(region,
                                                  ty::TypeAndMut {
                                                      ty: expr.ty,
                                                      mutbl: m,
                                                  }),
                                span: self.span,
                                kind: ExprKind::Borrow {
                                    region: region,
                                    borrow_kind: to_borrow_kind(m),
                                    arg: expr.to_ref(),
                                },
                            };
                            expr = Expr {
                                temp_lifetime: temp_lifetime,
                                ty: adjusted_ty,
                                span: self.span,
                                kind: ExprKind::Cast { source: expr.to_ref() },
                            };
                        }
                    }
                }

                if unsize {
                    expr = Expr {
                        temp_lifetime: temp_lifetime,
                        ty: adjusted_ty,
                        span: self.span,
                        kind: ExprKind::Unsize { source: expr.to_ref() },
                    };
                }
            }
        }

        // Next, wrap this up in the expr's scope.
        expr = Expr {
            temp_lifetime: temp_lifetime,
            ty: expr.ty,
            span: self.span,
            kind: ExprKind::Scope {
                extent: expr_extent,
                value: expr.to_ref(),
            },
        };

        // Finally, create a destruction scope, if any.
        if let Some(extent) = cx.tcx.region_maps.opt_destruction_extent(self.id) {
            expr = Expr {
                temp_lifetime: temp_lifetime,
                ty: expr.ty,
                span: self.span,
                kind: ExprKind::Scope {
                    extent: extent,
                    value: expr.to_ref(),
                },
            };
        }

        // OK, all done!
        expr
    }
}

fn make_mirror_unadjusted<'a, 'gcx, 'tcx>(cx: &mut Cx<'a, 'gcx, 'tcx>,
                                          expr: &'tcx hir::Expr)
                                          -> Expr<'tcx> {
    let expr_ty = cx.tables().expr_ty(expr);
    let temp_lifetime = cx.tcx.region_maps.temporary_scope(expr.id);

    let kind = match expr.node {
        // Here comes the interesting stuff:
        hir::ExprMethodCall(.., ref args) => {
            // Rewrite a.b(c) into UFCS form like Trait::b(a, c)
            let expr = method_callee(cx, expr, ty::MethodCall::expr(expr.id));
            let args = args.iter()
                .map(|e| e.to_ref())
                .collect();
            ExprKind::Call {
                ty: expr.ty,
                fun: expr.to_ref(),
                args: args,
            }
        }

        hir::ExprCall(ref fun, ref args) => {
            if cx.tables().is_method_call(expr.id) {
                // The callee is something implementing Fn, FnMut, or FnOnce.
                // Find the actual method implementation being called and
                // build the appropriate UFCS call expression with the
                // callee-object as expr parameter.

                // rewrite f(u, v) into FnOnce::call_once(f, (u, v))

                let method = method_callee(cx, expr, ty::MethodCall::expr(expr.id));

                let sig = match method.ty.sty {
                    ty::TyFnDef(.., fn_ty) => &fn_ty.sig,
                    _ => span_bug!(expr.span, "type of method is not an fn"),
                };

                let sig = cx.tcx
                    .no_late_bound_regions(sig)
                    .unwrap_or_else(|| span_bug!(expr.span, "method call has late-bound regions"));

                assert_eq!(sig.inputs().len(), 2);

                let tupled_args = Expr {
                    ty: sig.inputs()[1],
                    temp_lifetime: temp_lifetime,
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
                    let substs = cx.tables().node_id_item_substs(fun.id)
                        .unwrap_or_else(|| cx.tcx.intern_substs(&[]));
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
                        adt_def: adt_def,
                        substs: substs,
                        variant_index: index,
                        fields: field_refs,
                        base: None,
                    }
                } else {
                    ExprKind::Call {
                        ty: cx.tables().node_id_to_type(fun.id),
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
                region: region,
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
            if cx.tables().is_method_call(expr.id) {
                let pass_args = if op.node.is_by_value() {
                    PassArgs::ByValue
                } else {
                    PassArgs::ByRef
                };
                overloaded_operator(cx,
                                    expr,
                                    ty::MethodCall::expr(expr.id),
                                    pass_args,
                                    lhs.to_ref(),
                                    vec![rhs])
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
            if cx.tables().is_method_call(expr.id) {
                let pass_args = if op.node.is_by_value() {
                    PassArgs::ByValue
                } else {
                    PassArgs::ByRef
                };
                overloaded_operator(cx,
                                    expr,
                                    ty::MethodCall::expr(expr.id),
                                    pass_args,
                                    lhs.to_ref(),
                                    vec![rhs])
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
                            op: op,
                            lhs: lhs.to_ref(),
                            rhs: rhs.to_ref(),
                        }
                    }
                }
            }
        }

        hir::ExprIndex(ref lhs, ref index) => {
            if cx.tables().is_method_call(expr.id) {
                overloaded_lvalue(cx,
                                  expr,
                                  ty::MethodCall::expr(expr.id),
                                  PassArgs::ByValue,
                                  lhs.to_ref(),
                                  vec![index])
            } else {
                ExprKind::Index {
                    lhs: lhs.to_ref(),
                    index: index.to_ref(),
                }
            }
        }

        hir::ExprUnary(hir::UnOp::UnDeref, ref arg) => {
            if cx.tables().is_method_call(expr.id) {
                overloaded_lvalue(cx,
                                  expr,
                                  ty::MethodCall::expr(expr.id),
                                  PassArgs::ByValue,
                                  arg.to_ref(),
                                  vec![])
            } else {
                ExprKind::Deref { arg: arg.to_ref() }
            }
        }

        hir::ExprUnary(hir::UnOp::UnNot, ref arg) => {
            if cx.tables().is_method_call(expr.id) {
                overloaded_operator(cx,
                                    expr,
                                    ty::MethodCall::expr(expr.id),
                                    PassArgs::ByValue,
                                    arg.to_ref(),
                                    vec![])
            } else {
                ExprKind::Unary {
                    op: UnOp::Not,
                    arg: arg.to_ref(),
                }
            }
        }

        hir::ExprUnary(hir::UnOp::UnNeg, ref arg) => {
            if cx.tables().is_method_call(expr.id) {
                overloaded_operator(cx,
                                    expr,
                                    ty::MethodCall::expr(expr.id),
                                    PassArgs::ByValue,
                                    arg.to_ref(),
                                    vec![])
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
                                substs: substs,
                                fields: field_refs,
                                base: base.as_ref().map(|base| {
                                    FruInfo {
                                        base: base.to_ref(),
                                        field_types: cx.tables().fru_field_types[&expr.id].clone(),
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
                                        substs: substs,
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
            let (def_id, substs) = match closure_ty.sty {
                ty::TyClosure(def_id, substs) => (def_id, substs),
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
                substs: substs,
                upvars: upvars,
            }
        }

        hir::ExprPath(ref qpath) => {
            let def = cx.tables().qpath_def(qpath, expr.id);
            convert_path_expr(cx, expr, def)
        }

        hir::ExprInlineAsm(ref asm, ref outputs, ref inputs) => {
            ExprKind::InlineAsm {
                asm: asm,
                outputs: outputs.to_ref(),
                inputs: inputs.to_ref(),
            }
        }

        // Now comes the rote stuff:
        hir::ExprRepeat(ref v, count) => {
            let tcx = cx.tcx.global_tcx();
            let c = &cx.tcx.map.body(count).value;
            let count = match ConstContext::new(tcx, count).eval(c, EvalHint::ExprTypeChecked) {
                Ok(ConstVal::Integral(ConstInt::Usize(u))) => u,
                Ok(other) => bug!("constant evaluation of repeat count yielded {:?}", other),
                Err(s) => fatal_const_eval_err(tcx, &s, c.span, "expression")
            };

            ExprKind::Repeat {
                value: v.to_ref(),
                count: TypedConstVal {
                    ty: cx.tcx.types.usize,
                    span: c.span,
                    value: count
                }
            }
        }
        hir::ExprRet(ref v) => ExprKind::Return { value: v.to_ref() },
        hir::ExprBreak(label, ref value) => {
            ExprKind::Break {
                label: label.map(|label| cx.tcx.region_maps.node_extent(label.loop_id)),
                value: value.to_ref(),
            }
        }
        hir::ExprAgain(label) => {
            ExprKind::Continue {
                label: label.map(|label| cx.tcx.region_maps.node_extent(label.loop_id)),
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
                then: block::to_expr_ref(cx, then),
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
            if let Some(&TyCastKind::CoercionCast) = cx.tcx.cast_kinds.borrow().get(&source.id) {
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
                value_extents: cx.tcx.region_maps.node_extent(value.id),
            }
        }
        hir::ExprArray(ref fields) => ExprKind::Array { fields: fields.to_ref() },
        hir::ExprTup(ref fields) => ExprKind::Tuple { fields: fields.to_ref() },
    };

    Expr {
        temp_lifetime: temp_lifetime,
        ty: expr_ty,
        span: expr.span,
        kind: kind,
    }
}

fn method_callee<'a, 'gcx, 'tcx>(cx: &mut Cx<'a, 'gcx, 'tcx>,
                                 expr: &hir::Expr,
                                 method_call: ty::MethodCall)
                                 -> Expr<'tcx> {
    let callee = cx.tables().method_map[&method_call];
    let temp_lifetime = cx.tcx.region_maps.temporary_scope(expr.id);
    Expr {
        temp_lifetime: temp_lifetime,
        ty: callee.ty,
        span: expr.span,
        kind: ExprKind::Literal {
            literal: Literal::Item {
                def_id: callee.def_id,
                substs: callee.substs,
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
        patterns: arm.pats.iter().map(|p| Pattern::from_hir(cx.tcx, cx.tables(), p)).collect(),
        guard: arm.guard.to_ref(),
        body: arm.body.to_ref(),
    }
}

fn convert_path_expr<'a, 'gcx, 'tcx>(cx: &mut Cx<'a, 'gcx, 'tcx>,
                                     expr: &'tcx hir::Expr,
                                     def: Def)
                                     -> ExprKind<'tcx> {
    let substs = cx.tables().node_id_item_substs(expr.id)
        .unwrap_or_else(|| cx.tcx.intern_substs(&[]));
    let def_id = match def {
        // A regular function, constructor function or a constant.
        Def::Fn(def_id) |
        Def::Method(def_id) |
        Def::StructCtor(def_id, CtorKind::Fn) |
        Def::VariantCtor(def_id, CtorKind::Fn) |
        Def::Const(def_id) |
        Def::AssociatedConst(def_id) => def_id,

        Def::StructCtor(def_id, CtorKind::Const) |
        Def::VariantCtor(def_id, CtorKind::Const) => {
            match cx.tables().node_id_to_type(expr.id).sty {
                // A unit struct/variant which is used as a value.
                // We return a completely different ExprKind here to account for this special case.
                ty::TyAdt(adt_def, substs) => {
                    return ExprKind::Adt {
                        adt_def: adt_def,
                        variant_index: adt_def.variant_index_with_id(def_id),
                        substs: substs,
                        fields: vec![],
                        base: None,
                    }
                }
                ref sty => bug!("unexpected sty: {:?}", sty),
            }
        }

        Def::Static(node_id, _) => return ExprKind::StaticRef { id: node_id },

        Def::Local(..) | Def::Upvar(..) => return convert_var(cx, expr, def),

        _ => span_bug!(expr.span, "def `{:?}` not yet implemented", def),
    };
    ExprKind::Literal {
        literal: Literal::Item {
            def_id: def_id,
            substs: substs,
        },
    }
}

fn convert_var<'a, 'gcx, 'tcx>(cx: &mut Cx<'a, 'gcx, 'tcx>,
                               expr: &'tcx hir::Expr,
                               def: Def)
                               -> ExprKind<'tcx> {
    let temp_lifetime = cx.tcx.region_maps.temporary_scope(expr.id);

    match def {
        Def::Local(def_id) => {
            let node_id = cx.tcx.map.as_local_node_id(def_id).unwrap();
            ExprKind::VarRef { id: node_id }
        }

        Def::Upvar(def_id, index, closure_expr_id) => {
            let id_var = cx.tcx.map.as_local_node_id(def_id).unwrap();
            debug!("convert_var(upvar({:?}, {:?}, {:?}))",
                   id_var,
                   index,
                   closure_expr_id);
            let var_ty = cx.tables().node_id_to_type(id_var);

            let body_id = match cx.tcx.map.find(closure_expr_id) {
                Some(map::NodeExpr(expr)) => {
                    match expr.node {
                        hir::ExprClosure(.., body, _) => body.node_id,
                        _ => {
                            span_bug!(expr.span, "closure expr is not a closure expr");
                        }
                    }
                }
                _ => {
                    span_bug!(expr.span, "ast-map has garbage for closure expr");
                }
            };

            // FIXME free regions in closures are not right
            let closure_ty = cx.tables().node_id_to_type(closure_expr_id);

            // FIXME we're just hard-coding the idea that the
            // signature will be &self or &mut self and hence will
            // have a bound region with number 0
            let region = ty::Region::ReFree(ty::FreeRegion {
                scope: cx.tcx.region_maps.node_extent(body_id),
                bound_region: ty::BoundRegion::BrAnon(0),
            });
            let region = cx.tcx.mk_region(region);

            let self_expr = match cx.tcx.closure_kind(cx.tcx.map.local_def_id(closure_expr_id)) {
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
                                    temp_lifetime: temp_lifetime,
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
                        temp_lifetime: temp_lifetime,
                        span: expr.span,
                        kind: ExprKind::Deref {
                            arg: Expr {
                                    ty: ref_closure_ty,
                                    temp_lifetime: temp_lifetime,
                                    span: expr.span,
                                    kind: ExprKind::SelfRef,
                                }
                                .to_ref(),
                        },
                    }
                }
                ty::ClosureKind::FnOnce => {
                    Expr {
                        ty: closure_ty,
                        temp_lifetime: temp_lifetime,
                        span: expr.span,
                        kind: ExprKind::SelfRef,
                    }
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
                var_id: id_var,
                closure_expr_id: closure_expr_id,
            };
            let upvar_capture = match cx.tables().upvar_capture(upvar_id) {
                Some(c) => c,
                None => {
                    span_bug!(expr.span, "no upvar_capture for {:?}", upvar_id);
                }
            };
            match upvar_capture {
                ty::UpvarCapture::ByValue => field_kind,
                ty::UpvarCapture::ByRef(borrow) => {
                    ExprKind::Deref {
                        arg: Expr {
                                temp_lifetime: temp_lifetime,
                                ty: cx.tcx.mk_ref(borrow.region,
                                                  ty::TypeAndMut {
                                                      ty: var_ty,
                                                      mutbl: borrow.kind.to_mutbl_lossy(),
                                                  }),
                                span: expr.span,
                                kind: field_kind,
                            }
                            .to_ref(),
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

enum PassArgs {
    ByValue,
    ByRef,
}

fn overloaded_operator<'a, 'gcx, 'tcx>(cx: &mut Cx<'a, 'gcx, 'tcx>,
                                       expr: &'tcx hir::Expr,
                                       method_call: ty::MethodCall,
                                       pass_args: PassArgs,
                                       receiver: ExprRef<'tcx>,
                                       args: Vec<&'tcx P<hir::Expr>>)
                                       -> ExprKind<'tcx> {
    // the receiver has all the adjustments that are needed, so we can
    // just push a reference to it
    let mut argrefs = vec![receiver];

    // the arguments, unfortunately, do not, so if this is a ByRef
    // operator, we have to gin up the autorefs (but by value is easy)
    match pass_args {
        PassArgs::ByValue => argrefs.extend(args.iter().map(|arg| arg.to_ref())),

        PassArgs::ByRef => {
            let region = cx.tcx.node_scope_region(expr.id);
            let temp_lifetime = cx.tcx.region_maps.temporary_scope(expr.id);
            argrefs.extend(args.iter()
                .map(|arg| {
                    let arg_ty = cx.tables().expr_ty_adjusted(arg);
                    let adjusted_ty = cx.tcx.mk_ref(region,
                                                    ty::TypeAndMut {
                                                        ty: arg_ty,
                                                        mutbl: hir::MutImmutable,
                                                    });
                    Expr {
                            temp_lifetime: temp_lifetime,
                            ty: adjusted_ty,
                            span: expr.span,
                            kind: ExprKind::Borrow {
                                region: region,
                                borrow_kind: BorrowKind::Shared,
                                arg: arg.to_ref(),
                            },
                        }
                        .to_ref()
                }))
        }
    }

    // now create the call itself
    let fun = method_callee(cx, expr, method_call);
    ExprKind::Call {
        ty: fun.ty,
        fun: fun.to_ref(),
        args: argrefs,
    }
}

fn overloaded_lvalue<'a, 'gcx, 'tcx>(cx: &mut Cx<'a, 'gcx, 'tcx>,
                                     expr: &'tcx hir::Expr,
                                     method_call: ty::MethodCall,
                                     pass_args: PassArgs,
                                     receiver: ExprRef<'tcx>,
                                     args: Vec<&'tcx P<hir::Expr>>)
                                     -> ExprKind<'tcx> {
    // For an overloaded *x or x[y] expression of type T, the method
    // call returns an &T and we must add the deref so that the types
    // line up (this is because `*x` and `x[y]` represent lvalues):

    // to find the type &T of the content returned by the method;
    let ref_ty = cx.tables().method_map[&method_call].ty.fn_ret();
    let ref_ty = cx.tcx.no_late_bound_regions(&ref_ty).unwrap();
    // callees always have all late-bound regions fully instantiated,

    // construct the complete expression `foo()` for the overloaded call,
    // which will yield the &T type
    let temp_lifetime = cx.tcx.region_maps.temporary_scope(expr.id);
    let ref_kind = overloaded_operator(cx, expr, method_call, pass_args, receiver, args);
    let ref_expr = Expr {
        temp_lifetime: temp_lifetime,
        ty: ref_ty,
        span: expr.span,
        kind: ref_kind,
    };

    // construct and return a deref wrapper `*foo()`
    ExprKind::Deref { arg: ref_expr.to_ref() }
}

fn capture_freevar<'a, 'gcx, 'tcx>(cx: &mut Cx<'a, 'gcx, 'tcx>,
                                   closure_expr: &'tcx hir::Expr,
                                   freevar: &hir::Freevar,
                                   freevar_ty: Ty<'tcx>)
                                   -> ExprRef<'tcx> {
    let id_var = cx.tcx.map.as_local_node_id(freevar.def.def_id()).unwrap();
    let upvar_id = ty::UpvarId {
        var_id: id_var,
        closure_expr_id: closure_expr.id,
    };
    let upvar_capture = cx.tables().upvar_capture(upvar_id).unwrap();
    let temp_lifetime = cx.tcx.region_maps.temporary_scope(closure_expr.id);
    let var_ty = cx.tables().node_id_to_type(id_var);
    let captured_var = Expr {
        temp_lifetime: temp_lifetime,
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
                    temp_lifetime: temp_lifetime,
                    ty: freevar_ty,
                    span: closure_expr.span,
                    kind: ExprKind::Borrow {
                        region: upvar_borrow.region,
                        borrow_kind: borrow_kind,
                        arg: captured_var.to_ref(),
                    },
                }
                .to_ref()
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
