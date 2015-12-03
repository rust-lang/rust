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
use rustc_data_structures::fnv::FnvHashMap;
use hair::cx::Cx;
use hair::cx::block;
use hair::cx::to_ref::ToRef;
use rustc::front::map;
use rustc::middle::def;
use rustc::middle::region::CodeExtent;
use rustc::middle::pat_util;
use rustc::middle::ty::{self, VariantDef, Ty};
use rustc::mir::repr::*;
use rustc_front::hir;
use rustc_front::util as hir_util;
use syntax::parse::token;
use syntax::ptr::P;

impl<'tcx> Mirror<'tcx> for &'tcx hir::Expr {
    type Output = Expr<'tcx>;

    fn make_mirror<'a>(self, cx: &mut Cx<'a, 'tcx>) -> Expr<'tcx> {
        debug!("Expr::make_mirror(): id={}, span={:?}", self.id, self.span);

        let expr_ty = cx.tcx.expr_ty(self); // note: no adjustments (yet)!

        let kind = match self.node {
            // Here comes the interesting stuff:
            hir::ExprMethodCall(_, _, ref args) => {
                // Rewrite a.b(c) into UFCS form like Trait::b(a, c)
                let expr = method_callee(cx, self, ty::MethodCall::expr(self.id));
                let args = args.iter()
                               .map(|e| e.to_ref())
                               .collect();
                ExprKind::Call {
                    fun: expr.to_ref(),
                    args: args,
                }
            }

            hir::ExprAddrOf(mutbl, ref expr) => {
                let region = match expr_ty.sty {
                    ty::TyRef(r, _) => r,
                    _ => cx.tcx.sess.span_bug(expr.span, "type of & not region"),
                };
                ExprKind::Borrow {
                    region: *region,
                    borrow_kind: to_borrow_kind(mutbl),
                    arg: expr.to_ref(),
                }
            }

            hir::ExprBlock(ref blk) => {
                ExprKind::Block { body: &**blk }
            }

            hir::ExprAssign(ref lhs, ref rhs) => {
                ExprKind::Assign {
                    lhs: lhs.to_ref(),
                    rhs: rhs.to_ref(),
                }
            }

            hir::ExprAssignOp(op, ref lhs, ref rhs) => {
                let op = bin_op(op.node);
                ExprKind::AssignOp {
                    op: op,
                    lhs: lhs.to_ref(),
                    rhs: rhs.to_ref(),
                }
            }

            hir::ExprLit(..) => ExprKind::Literal {
                literal: cx.const_eval_literal(self)
            },

            hir::ExprBinary(op, ref lhs, ref rhs) => {
                if cx.tcx.is_method_call(self.id) {
                    let pass_args = if hir_util::is_by_value_binop(op.node) {
                        PassArgs::ByValue
                    } else {
                        PassArgs::ByRef
                    };
                    overloaded_operator(cx, self, ty::MethodCall::expr(self.id),
                                        pass_args, lhs.to_ref(), vec![rhs])
                } else {
                    // FIXME overflow
                    match op.node {
                        hir::BinOp_::BiAnd => {
                            ExprKind::LogicalOp {
                                op: LogicalOp::And,
                                lhs: lhs.to_ref(),
                                rhs: rhs.to_ref(),
                            }
                        }
                        hir::BinOp_::BiOr => {
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
                if cx.tcx.is_method_call(self.id) {
                    overloaded_lvalue(cx, self, ty::MethodCall::expr(self.id),
                                      PassArgs::ByValue, lhs.to_ref(), vec![index])
                } else {
                    ExprKind::Index {
                        lhs: lhs.to_ref(),
                        index: index.to_ref(),
                    }
                }
            }

            hir::ExprUnary(hir::UnOp::UnDeref, ref arg) => {
                if cx.tcx.is_method_call(self.id) {
                    overloaded_lvalue(cx, self, ty::MethodCall::expr(self.id),
                                      PassArgs::ByValue, arg.to_ref(), vec![])
                } else {
                    ExprKind::Deref { arg: arg.to_ref() }
                }
            }

            hir::ExprUnary(op, ref arg) => {
                if cx.tcx.is_method_call(self.id) {
                    overloaded_operator(cx, self, ty::MethodCall::expr(self.id),
                                        PassArgs::ByValue, arg.to_ref(), vec![])
                } else {
                    // FIXME overflow
                    let op = match op {
                        hir::UnOp::UnNot => UnOp::Not,
                        hir::UnOp::UnNeg => UnOp::Neg,
                        hir::UnOp::UnDeref => {
                            cx.tcx.sess.span_bug(
                                self.span,
                                "UnDeref should have been handled elsewhere");
                        }
                    };
                    ExprKind::Unary {
                        op: op,
                        arg: arg.to_ref(),
                    }
                }
            }

            hir::ExprStruct(_, ref fields, ref base) => {
                match expr_ty.sty {
                    ty::TyStruct(adt, substs) => {
                        let field_refs = field_refs(&adt.variants[0], fields);
                        ExprKind::Adt {
                            adt_def: adt,
                            variant_index: 0,
                            substs: substs,
                            fields: field_refs,
                            base: base.to_ref(),
                        }
                    }
                    ty::TyEnum(adt, substs) => {
                        match cx.tcx.def_map.borrow()[&self.id].full_def() {
                            def::DefVariant(enum_id, variant_id, _) => {
                                debug_assert!(adt.did == enum_id);
                                let index = adt.variant_index_with_id(variant_id);
                                let field_refs = field_refs(&adt.variants[index], fields);
                                ExprKind::Adt {
                                    adt_def: adt,
                                    variant_index: index,
                                    substs: substs,
                                    fields: field_refs,
                                    base: base.to_ref(),
                                }
                            }
                            ref def => {
                                cx.tcx.sess.span_bug(
                                    self.span,
                                    &format!("unexpected def: {:?}", def));
                            }
                        }
                    }
                    _ => {
                        cx.tcx.sess.span_bug(
                            self.span,
                            &format!("unexpected type for struct literal: {:?}", expr_ty));
                    }
                }
            }

            hir::ExprClosure(..) => {
                let closure_ty = cx.tcx.expr_ty(self);
                let (def_id, substs) = match closure_ty.sty {
                    ty::TyClosure(def_id, ref substs) => (def_id, substs),
                    _ => {
                        cx.tcx.sess.span_bug(self.span,
                                             &format!("closure expr w/o closure type: {:?}",
                                                      closure_ty));
                    }
                };
                let upvars = cx.tcx.with_freevars(self.id, |freevars| {
                    freevars.iter()
                            .enumerate()
                            .map(|(i, fv)| capture_freevar(cx, self, fv, substs.upvar_tys[i]))
                            .collect()
                });
                ExprKind::Closure {
                    closure_id: def_id,
                    substs: &**substs,
                    upvars: upvars,
                }
            }

            hir::ExprRange(ref start, ref end) => {
                let range_ty = cx.tcx.expr_ty(self);
                let (adt_def, substs) = match range_ty.sty {
                    ty::TyStruct(adt_def, substs) => (adt_def, substs),
                    _ => {
                        cx.tcx.sess.span_bug(self.span, "unexpanded ast");
                    }
                };

                let field_expr_ref = |s: &'tcx P<hir::Expr>, name: &str| {
                    let name = token::intern(name);
                    let index = adt_def.variants[0].index_of_field_named(name).unwrap();
                    FieldExprRef { name: Field::new(index), expr: s.to_ref() }
                };

                let start_field = start.as_ref()
                                       .into_iter()
                                       .map(|s| field_expr_ref(s, "start"));

                let end_field = end.as_ref()
                                   .into_iter()
                                   .map(|e| field_expr_ref(e, "end"));

                ExprKind::Adt {
                    adt_def: adt_def,
                    variant_index: 0,
                    substs: substs,
                    fields: start_field.chain(end_field).collect(),
                    base: None,
                }
            }

            hir::ExprPath(..) => {
                convert_path_expr(cx, self)
            }

            hir::ExprInlineAsm(ref asm) => {
                ExprKind::InlineAsm { asm: asm }
            }

            // Now comes the rote stuff:

            hir::ExprRepeat(ref v, ref c) => ExprKind::Repeat {
                value: v.to_ref(),
                count: Expr {
                    ty: cx.tcx.expr_ty(c),
                    temp_lifetime: None,
                    span: c.span,
                    kind: ExprKind::Literal {
                        literal: cx.const_eval_literal(c)
                    }
                }.to_ref()
            },
            hir::ExprRet(ref v) =>
                ExprKind::Return { value: v.to_ref() },
            hir::ExprBreak(label) =>
                ExprKind::Break { label: label.map(|_| loop_label(cx, self)) },
            hir::ExprAgain(label) =>
                ExprKind::Continue { label: label.map(|_| loop_label(cx, self)) },
            hir::ExprMatch(ref discr, ref arms, _) =>
                ExprKind::Match { discriminant: discr.to_ref(),
                                  arms: arms.iter().map(|a| convert_arm(cx, a)).collect() },
            hir::ExprIf(ref cond, ref then, ref otherwise) =>
                ExprKind::If { condition: cond.to_ref(),
                               then: block::to_expr_ref(cx, then),
                               otherwise: otherwise.to_ref() },
            hir::ExprWhile(ref cond, ref body, _) =>
                ExprKind::Loop { condition: Some(cond.to_ref()),
                                 body: block::to_expr_ref(cx, body) },
            hir::ExprLoop(ref body, _) =>
                ExprKind::Loop { condition: None,
                                 body: block::to_expr_ref(cx, body) },
            hir::ExprField(ref source, name) => {
                let index = match cx.tcx.expr_ty_adjusted(source).sty {
                    ty::TyStruct(adt_def, _) =>
                        adt_def.variants[0].index_of_field_named(name.node),
                    ref ty =>
                        cx.tcx.sess.span_bug(
                            self.span,
                            &format!("field of non-struct: {:?}", ty)),
                };
                let index = index.unwrap_or_else(|| {
                    cx.tcx.sess.span_bug(
                        self.span,
                        &format!("no index found for field `{}`", name.node));
                });
                ExprKind::Field { lhs: source.to_ref(), name: Field::new(index) }
            }
            hir::ExprTupField(ref source, index) =>
                ExprKind::Field { lhs: source.to_ref(),
                                  name: Field::new(index.node as usize) },
            hir::ExprCast(ref source, _) =>
                ExprKind::Cast { source: source.to_ref() },
            hir::ExprType(ref source, _) =>
                return source.make_mirror(cx),
            hir::ExprBox(ref value) =>
                ExprKind::Box { value: value.to_ref() },
            hir::ExprVec(ref fields) =>
                ExprKind::Vec { fields: fields.to_ref() },
            hir::ExprTup(ref fields) =>
                ExprKind::Tuple { fields: fields.to_ref() },
            hir::ExprCall(ref fun, ref args) =>
                ExprKind::Call { fun: fun.to_ref(), args: args.to_ref() },
        };

        let temp_lifetime = cx.tcx.region_maps.temporary_scope(self.id);
        let expr_extent = cx.tcx.region_maps.node_extent(self.id);

        let mut expr = Expr {
            temp_lifetime: temp_lifetime,
            ty: expr_ty,
            span: self.span,
            kind: kind,
        };

        // Now apply adjustments, if any.
        match cx.tcx.tables.borrow().adjustments.get(&self.id) {
            None => {}
            Some(&ty::adjustment::AdjustReifyFnPointer) => {
                let adjusted_ty = cx.tcx.expr_ty_adjusted(self);
                expr = Expr {
                    temp_lifetime: temp_lifetime,
                    ty: adjusted_ty,
                    span: self.span,
                    kind: ExprKind::ReifyFnPointer { source: expr.to_ref() },
                };
            }
            Some(&ty::adjustment::AdjustUnsafeFnPointer) => {
                let adjusted_ty = cx.tcx.expr_ty_adjusted(self);
                expr = Expr {
                    temp_lifetime: temp_lifetime,
                    ty: adjusted_ty,
                    span: self.span,
                    kind: ExprKind::UnsafeFnPointer { source: expr.to_ref() },
                };
            }
            Some(&ty::adjustment::AdjustDerefRef(ref adj)) => {
                for i in 0..adj.autoderefs {
                    let i = i as u32;
                    let adjusted_ty =
                        expr.ty.adjust_for_autoderef(
                            cx.tcx,
                            self.id,
                            self.span,
                            i,
                            |mc| cx.tcx.tables.borrow().method_map.get(&mc).map(|m| m.ty));
                    let kind = if cx.tcx.is_overloaded_autoderef(self.id, i) {
                        overloaded_lvalue(cx, self, ty::MethodCall::autoderef(self.id, i),
                                          PassArgs::ByValue, expr.to_ref(), vec![])
                    } else {
                        ExprKind::Deref { arg: expr.to_ref() }
                    };
                    expr = Expr {
                        temp_lifetime: temp_lifetime,
                        ty: adjusted_ty,
                        span: self.span,
                        kind: kind,
                    };
                }

                if let Some(autoref) = adj.autoref {
                    let adjusted_ty = expr.ty.adjust_for_autoref(cx.tcx, Some(autoref));
                    match autoref {
                        ty::adjustment::AutoPtr(r, m) => {
                            expr = Expr {
                                temp_lifetime: temp_lifetime,
                                ty: adjusted_ty,
                                span: self.span,
                                kind: ExprKind::Borrow {
                                    region: *r,
                                    borrow_kind: to_borrow_kind(m),
                                    arg: expr.to_ref(),
                                },
                            };
                        }
                        ty::adjustment::AutoUnsafe(m) => {
                            // Convert this to a suitable `&foo` and
                            // then an unsafe coercion. Limit the region to be just this
                            // expression.
                            let region = ty::ReScope(expr_extent);
                            let region = cx.tcx.mk_region(region);
                            expr = Expr {
                                temp_lifetime: temp_lifetime,
                                ty: cx.tcx.mk_ref(region, ty::TypeAndMut { ty: expr.ty, mutbl: m }),
                                span: self.span,
                                kind: ExprKind::Borrow {
                                    region: *region,
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

                if let Some(target) = adj.unsize {
                    expr = Expr {
                        temp_lifetime: temp_lifetime,
                        ty: target,
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

fn method_callee<'a, 'tcx: 'a>(cx: &mut Cx<'a, 'tcx>,
                               expr: &hir::Expr,
                               method_call: ty::MethodCall)
                               -> Expr<'tcx> {
    let tables = cx.tcx.tables.borrow();
    let callee = &tables.method_map[&method_call];
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

fn convert_arm<'a, 'tcx: 'a>(cx: &mut Cx<'a, 'tcx>, arm: &'tcx hir::Arm) -> Arm<'tcx> {
    let mut map;
    let opt_map = if arm.pats.len() == 1 {
        None
    } else {
        map = FnvHashMap();
        pat_util::pat_bindings(&cx.tcx.def_map, &arm.pats[0], |_, p_id, _, path| {
            map.insert(path.node, p_id);
        });
        Some(&map)
    };

    Arm {
        patterns: arm.pats.iter().map(|p| cx.refutable_pat(opt_map, p)).collect(),
        guard: arm.guard.to_ref(),
        body: arm.body.to_ref(),
    }
}

fn convert_path_expr<'a, 'tcx: 'a>(cx: &mut Cx<'a, 'tcx>, expr: &'tcx hir::Expr) -> ExprKind<'tcx> {
    let substs = cx.tcx.mk_substs(cx.tcx.node_id_item_substs(expr.id).substs);
    match cx.tcx.def_map.borrow()[&expr.id].full_def() {
        def::DefVariant(_, def_id, false) |
        def::DefStruct(def_id) |
        def::DefFn(def_id, _) |
        def::DefConst(def_id) |
        def::DefMethod(def_id) |
        def::DefAssociatedConst(def_id) =>
            ExprKind::Literal {
                literal: Literal::Item { def_id: def_id, substs: substs }
            },

        def::DefStatic(node_id, _) =>
            ExprKind::StaticRef {
                id: node_id,
            },

        def @ def::DefLocal(..) |
        def @ def::DefUpvar(..) =>
            convert_var(cx, expr, def),

        def =>
            cx.tcx.sess.span_bug(
                expr.span,
                &format!("def `{:?}` not yet implemented", def)),
    }
}

fn convert_var<'a, 'tcx: 'a>(cx: &mut Cx<'a, 'tcx>,
                             expr: &'tcx hir::Expr,
                             def: def::Def)
                             -> ExprKind<'tcx> {
    let temp_lifetime = cx.tcx.region_maps.temporary_scope(expr.id);

    match def {
        def::DefLocal(_, node_id) => {
            ExprKind::VarRef {
                id: node_id,
            }
        }

        def::DefUpvar(_, id_var, index, closure_expr_id) => {
            debug!("convert_var(upvar({:?}, {:?}, {:?}))", id_var, index, closure_expr_id);
            let var_ty = cx.tcx.node_id_to_type(id_var);

            let body_id = match cx.tcx.map.find(closure_expr_id) {
                Some(map::NodeExpr(expr)) => {
                    match expr.node {
                        hir::ExprClosure(_, _, ref body) => body.id,
                        _ => {
                            cx.tcx.sess.span_bug(expr.span, "closure expr is not a closure expr");
                        }
                    }
                }
                _ => {
                    cx.tcx.sess.span_bug(expr.span, "ast-map has garbage for closure expr");
                }
            };

            // FIXME free regions in closures are not right
            let closure_ty = cx.tcx.node_id_to_type(closure_expr_id);

            // FIXME we're just hard-coding the idea that the
            // signature will be &self or &mut self and hence will
            // have a bound region with number 0
            let region = ty::Region::ReFree(ty::FreeRegion {
                scope: cx.tcx.region_maps.node_extent(body_id),
                bound_region: ty::BoundRegion::BrAnon(0),
            });
            let region = cx.tcx.mk_region(region);

            let self_expr = match cx.tcx.closure_kind(cx.tcx.map.local_def_id(closure_expr_id)) {
                ty::ClosureKind::FnClosureKind => {
                    let ref_closure_ty =
                        cx.tcx.mk_ref(region,
                                   ty::TypeAndMut { ty: closure_ty,
                                                    mutbl: hir::MutImmutable });
                    Expr {
                        ty: closure_ty,
                        temp_lifetime: temp_lifetime,
                        span: expr.span,
                        kind: ExprKind::Deref {
                            arg: Expr {
                                ty: ref_closure_ty,
                                temp_lifetime: temp_lifetime,
                                span: expr.span,
                                kind: ExprKind::SelfRef
                            }.to_ref()
                        }
                    }
                }
                ty::ClosureKind::FnMutClosureKind => {
                    let ref_closure_ty =
                        cx.tcx.mk_ref(region,
                                   ty::TypeAndMut { ty: closure_ty,
                                                    mutbl: hir::MutMutable });
                    Expr {
                        ty: closure_ty,
                        temp_lifetime: temp_lifetime,
                        span: expr.span,
                        kind: ExprKind::Deref {
                            arg: Expr {
                                ty: ref_closure_ty,
                                temp_lifetime: temp_lifetime,
                                span: expr.span,
                                kind: ExprKind::SelfRef
                            }.to_ref()
                        }
                    }
                }
                ty::ClosureKind::FnOnceClosureKind => {
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
            let upvar_capture = match cx.tcx.upvar_capture(upvar_id) {
                Some(c) => c,
                None => {
                    cx.tcx.sess.span_bug(
                        expr.span,
                        &format!("no upvar_capture for {:?}", upvar_id));
                }
            };
            match upvar_capture {
                ty::UpvarCapture::ByValue => field_kind,
                ty::UpvarCapture::ByRef(_) => {
                    ExprKind::Deref {
                        arg: Expr {
                            temp_lifetime: temp_lifetime,
                            ty: var_ty,
                            span: expr.span,
                            kind: field_kind,
                        }.to_ref()
                    }
                }
            }
        }

        _ => cx.tcx.sess.span_bug(expr.span, "type of & not region"),
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
        _ => panic!("no equivalent for ast binop {:?}", op),
    }
}

enum PassArgs {
    ByValue,
    ByRef,
}

fn overloaded_operator<'a, 'tcx: 'a>(cx: &mut Cx<'a, 'tcx>,
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
        PassArgs::ByValue => {
            argrefs.extend(args.iter().map(|arg| arg.to_ref()))
        }

        PassArgs::ByRef => {
            let scope = cx.tcx.region_maps.node_extent(expr.id);
            let region = cx.tcx.mk_region(ty::ReScope(scope));
            let temp_lifetime = cx.tcx.region_maps.temporary_scope(expr.id);
            argrefs.extend(
                args.iter()
                    .map(|arg| {
                        let arg_ty = cx.tcx.expr_ty_adjusted(arg);
                        let adjusted_ty =
                            cx.tcx.mk_ref(region,
                                       ty::TypeAndMut { ty: arg_ty,
                                                        mutbl: hir::MutImmutable });
                        Expr {
                            temp_lifetime: temp_lifetime,
                            ty: adjusted_ty,
                            span: expr.span,
                            kind: ExprKind::Borrow { region: *region,
                                                     borrow_kind: BorrowKind::Shared,
                                                     arg: arg.to_ref() }
                        }.to_ref()
                    }))
        }
    }

    // now create the call itself
    let fun = method_callee(cx, expr, method_call);
    ExprKind::Call {
        fun: fun.to_ref(),
        args: argrefs,
    }
}

fn overloaded_lvalue<'a, 'tcx: 'a>(cx: &mut Cx<'a, 'tcx>,
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
    let tables = cx.tcx.tables.borrow();
    let callee = &tables.method_map[&method_call];
    let ref_ty = callee.ty.fn_ret();
    let ref_ty = cx.tcx.no_late_bound_regions(&ref_ty).unwrap().unwrap();
    //                                              1~~~~~   2~~~~~
    // (1) callees always have all late-bound regions fully instantiated,
    // (2) overloaded methods don't return `!`

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

fn capture_freevar<'a, 'tcx: 'a>(cx: &mut Cx<'a, 'tcx>,
                                 closure_expr: &'tcx hir::Expr,
                                 freevar: &ty::Freevar,
                                 freevar_ty: Ty<'tcx>)
                                 -> ExprRef<'tcx> {
    let id_var = freevar.def.var_id();
    let upvar_id = ty::UpvarId {
        var_id: id_var,
        closure_expr_id: closure_expr.id,
    };
    let upvar_capture = cx.tcx.upvar_capture(upvar_id).unwrap();
    let temp_lifetime = cx.tcx.region_maps.temporary_scope(closure_expr.id);
    let var_ty = cx.tcx.node_id_to_type(id_var);
    let captured_var = Expr {
        temp_lifetime: temp_lifetime,
        ty: var_ty,
        span: closure_expr.span,
        kind: convert_var(cx, closure_expr, freevar.def),
    };
    match upvar_capture {
        ty::UpvarCapture::ByValue => {
            captured_var.to_ref()
        }
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
                kind: ExprKind::Borrow { region: upvar_borrow.region,
                                         borrow_kind: borrow_kind,
                                         arg: captured_var.to_ref() }
            }.to_ref()
        }
    }
}

fn loop_label<'a, 'tcx: 'a>(cx: &mut Cx<'a, 'tcx>, expr: &'tcx hir::Expr) -> CodeExtent {
    match cx.tcx.def_map.borrow().get(&expr.id).map(|d| d.full_def()) {
        Some(def::DefLabel(loop_id)) => cx.tcx.region_maps.node_extent(loop_id),
        d => {
            cx.tcx.sess.span_bug(expr.span, &format!("loop scope resolved to {:?}", d));
        }
    }
}

fn field_refs<'tcx>(variant: VariantDef<'tcx>,
                    fields: &'tcx [hir::Field])
                    -> Vec<FieldExprRef<'tcx>>
{
    fields.iter()
          .map(|field| FieldExprRef {
              name: Field::new(variant.index_of_field_named(field.name.node).unwrap()),
              expr: field.expr.to_ref(),
          })
          .collect()
}
