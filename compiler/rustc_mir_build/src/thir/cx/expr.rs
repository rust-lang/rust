use crate::thir::cx::region::Scope;
use crate::thir::cx::Cx;
use crate::thir::util::UserAnnotatedTyHelpers;
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, CtorOf, DefKind, Res};
use rustc_index::vec::Idx;
use rustc_middle::hir::place::Place as HirPlace;
use rustc_middle::hir::place::PlaceBase as HirPlaceBase;
use rustc_middle::hir::place::ProjectionKind as HirProjectionKind;
use rustc_middle::middle::region;
use rustc_middle::mir::{self, BinOp, BorrowKind, Field, UnOp};
use rustc_middle::thir::*;
use rustc_middle::ty::adjustment::{
    Adjust, Adjustment, AutoBorrow, AutoBorrowMutability, PointerCast,
};
use rustc_middle::ty::subst::{InternalSubsts, SubstsRef};
use rustc_middle::ty::{
    self, AdtKind, InlineConstSubsts, InlineConstSubstsParts, ScalarInt, Ty, UpvarSubsts, UserType,
};
use rustc_span::def_id::DefId;
use rustc_span::Span;
use rustc_target::abi::VariantIdx;

impl<'tcx> Cx<'tcx> {
    pub(crate) fn mirror_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) -> ExprId {
        // `mirror_expr` is recursing very deep. Make sure the stack doesn't overflow.
        ensure_sufficient_stack(|| self.mirror_expr_inner(expr))
    }

    pub(crate) fn mirror_exprs(&mut self, exprs: &'tcx [hir::Expr<'tcx>]) -> Box<[ExprId]> {
        exprs.iter().map(|expr| self.mirror_expr_inner(expr)).collect()
    }

    #[instrument(level = "trace", skip(self, hir_expr))]
    pub(super) fn mirror_expr_inner(&mut self, hir_expr: &'tcx hir::Expr<'tcx>) -> ExprId {
        let temp_lifetime =
            self.rvalue_scopes.temporary_scope(self.region_scope_tree, hir_expr.hir_id.local_id);
        let expr_scope =
            region::Scope { id: hir_expr.hir_id.local_id, data: region::ScopeData::Node };

        trace!(?hir_expr.hir_id, ?hir_expr.span);

        let mut expr = self.make_mirror_unadjusted(hir_expr);

        let adjustment_span = match self.adjustment_span {
            Some((hir_id, span)) if hir_id == hir_expr.hir_id => Some(span),
            _ => None,
        };

        // Now apply adjustments, if any.
        for adjustment in self.typeck_results.expr_adjustments(hir_expr) {
            trace!(?expr, ?adjustment);
            let span = expr.span;
            expr =
                self.apply_adjustment(hir_expr, expr, adjustment, adjustment_span.unwrap_or(span));
        }

        // Next, wrap this up in the expr's scope.
        expr = Expr {
            temp_lifetime,
            ty: expr.ty,
            span: hir_expr.span,
            kind: ExprKind::Scope {
                region_scope: expr_scope,
                value: self.thir.exprs.push(expr),
                lint_level: LintLevel::Explicit(hir_expr.hir_id),
            },
        };

        // Finally, create a destruction scope, if any.
        if let Some(region_scope) =
            self.region_scope_tree.opt_destruction_scope(hir_expr.hir_id.local_id)
        {
            expr = Expr {
                temp_lifetime,
                ty: expr.ty,
                span: hir_expr.span,
                kind: ExprKind::Scope {
                    region_scope,
                    value: self.thir.exprs.push(expr),
                    lint_level: LintLevel::Inherited,
                },
            };
        }

        // OK, all done!
        self.thir.exprs.push(expr)
    }

    fn apply_adjustment(
        &mut self,
        hir_expr: &'tcx hir::Expr<'tcx>,
        mut expr: Expr<'tcx>,
        adjustment: &Adjustment<'tcx>,
        mut span: Span,
    ) -> Expr<'tcx> {
        let Expr { temp_lifetime, .. } = expr;

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
            if let ExprKind::Block { block } = expr.kind {
                if let Some(last_expr) = self.thir[block].expr {
                    span = self.thir[last_expr].span;
                    expr.span = span;
                }
            }
        };

        let kind = match adjustment.kind {
            Adjust::Pointer(PointerCast::Unsize) => {
                adjust_span(&mut expr);
                ExprKind::Pointer { cast: PointerCast::Unsize, source: self.thir.exprs.push(expr) }
            }
            Adjust::Pointer(cast) => ExprKind::Pointer { cast, source: self.thir.exprs.push(expr) },
            Adjust::NeverToAny => ExprKind::NeverToAny { source: self.thir.exprs.push(expr) },
            Adjust::Deref(None) => {
                adjust_span(&mut expr);
                ExprKind::Deref { arg: self.thir.exprs.push(expr) }
            }
            Adjust::Deref(Some(deref)) => {
                // We don't need to do call adjust_span here since
                // deref coercions always start with a built-in deref.
                let call = deref.method_call(self.tcx(), expr.ty);

                expr = Expr {
                    temp_lifetime,
                    ty: self
                        .tcx
                        .mk_ref(deref.region, ty::TypeAndMut { ty: expr.ty, mutbl: deref.mutbl }),
                    span,
                    kind: ExprKind::Borrow {
                        borrow_kind: deref.mutbl.to_borrow_kind(),
                        arg: self.thir.exprs.push(expr),
                    },
                };

                let expr = Box::new([self.thir.exprs.push(expr)]);

                self.overloaded_place(hir_expr, adjustment.target, Some(call), expr, deref.span)
            }
            Adjust::Borrow(AutoBorrow::Ref(_, m)) => ExprKind::Borrow {
                borrow_kind: m.to_borrow_kind(),
                arg: self.thir.exprs.push(expr),
            },
            Adjust::Borrow(AutoBorrow::RawPtr(mutability)) => {
                ExprKind::AddressOf { mutability, arg: self.thir.exprs.push(expr) }
            }
        };

        Expr { temp_lifetime, ty: adjustment.target, span, kind }
    }

    /// Lowers a cast expression.
    ///
    /// Dealing with user type annotations is left to the caller.
    fn mirror_expr_cast(
        &mut self,
        source: &'tcx hir::Expr<'tcx>,
        temp_lifetime: Option<Scope>,
        span: Span,
    ) -> ExprKind<'tcx> {
        let tcx = self.tcx;

        // Check to see if this cast is a "coercion cast", where the cast is actually done
        // using a coercion (or is a no-op).
        if self.typeck_results().is_coercion_cast(source.hir_id) {
            // Convert the lexpr to a vexpr.
            ExprKind::Use { source: self.mirror_expr(source) }
        } else if self.typeck_results().expr_ty(source).is_region_ptr() {
            // Special cased so that we can type check that the element
            // type of the source matches the pointed to type of the
            // destination.
            ExprKind::Pointer {
                source: self.mirror_expr(source),
                cast: PointerCast::ArrayToPointer,
            }
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

            let hir::ExprKind::Path(ref qpath) = source.kind else {
                return ExprKind::Cast { source: self.mirror_expr(source)};
            };

            let res = self.typeck_results().qpath_res(qpath, source.hir_id);
            let ty = self.typeck_results().node_type(source.hir_id);
            let ty::Adt(adt_def, substs) = ty.kind() else {
                return ExprKind::Cast { source: self.mirror_expr(source)};
            };

            let Res::Def(DefKind::Ctor(CtorOf::Variant, CtorKind::Const), variant_ctor_id) = res else {
                return ExprKind::Cast { source: self.mirror_expr(source)};
            };

            let idx = adt_def.variant_index_with_ctor_id(variant_ctor_id);
            let (discr_did, discr_offset) = adt_def.discriminant_def_for_variant(idx);

            use rustc_middle::ty::util::IntTypeExt;
            let ty = adt_def.repr().discr_type();
            let discr_ty = ty.to_ty(tcx);

            let param_env_ty = self.param_env.and(discr_ty);
            let size = tcx
                .layout_of(param_env_ty)
                .unwrap_or_else(|e| {
                    panic!("could not compute layout for {:?}: {:?}", param_env_ty, e)
                })
                .size;

            let lit = ScalarInt::try_from_uint(discr_offset as u128, size).unwrap();
            let kind = ExprKind::NonHirLiteral { lit, user_ty: None };
            let offset = self.thir.exprs.push(Expr { temp_lifetime, ty: discr_ty, span, kind });

            let source = match discr_did {
                // in case we are offsetting from a computed discriminant
                // and not the beginning of discriminants (which is always `0`)
                Some(did) => {
                    let kind = ExprKind::NamedConst { def_id: did, substs, user_ty: None };
                    let lhs =
                        self.thir.exprs.push(Expr { temp_lifetime, ty: discr_ty, span, kind });
                    let bin = ExprKind::Binary { op: BinOp::Add, lhs, rhs: offset };
                    self.thir.exprs.push(Expr {
                        temp_lifetime,
                        ty: discr_ty,
                        span: span,
                        kind: bin,
                    })
                }
                None => offset,
            };

            ExprKind::Cast { source }
        }
    }

    fn make_mirror_unadjusted(&mut self, expr: &'tcx hir::Expr<'tcx>) -> Expr<'tcx> {
        let tcx = self.tcx;
        let expr_ty = self.typeck_results().expr_ty(expr);
        let expr_span = expr.span;
        let temp_lifetime =
            self.rvalue_scopes.temporary_scope(self.region_scope_tree, expr.hir_id.local_id);

        let kind = match expr.kind {
            // Here comes the interesting stuff:
            hir::ExprKind::MethodCall(segment, ref args, fn_span) => {
                // Rewrite a.b(c) into UFCS form like Trait::b(a, c)
                let expr = self.method_callee(expr, segment.ident.span, None);
                // When we apply adjustments to the receiver, use the span of
                // the overall method call for better diagnostics. args[0]
                // is guaranteed to exist, since a method call always has a receiver.
                let old_adjustment_span = self.adjustment_span.replace((args[0].hir_id, expr_span));
                info!("Using method span: {:?}", expr.span);
                let args = self.mirror_exprs(args);
                self.adjustment_span = old_adjustment_span;
                ExprKind::Call {
                    ty: expr.ty,
                    fun: self.thir.exprs.push(expr),
                    args,
                    from_hir_call: true,
                    fn_span,
                }
            }

            hir::ExprKind::Call(ref fun, ref args) => {
                if self.typeck_results().is_method_call(expr) {
                    // The callee is something implementing Fn, FnMut, or FnOnce.
                    // Find the actual method implementation being called and
                    // build the appropriate UFCS call expression with the
                    // callee-object as expr parameter.

                    // rewrite f(u, v) into FnOnce::call_once(f, (u, v))

                    let method = self.method_callee(expr, fun.span, None);

                    let arg_tys = args.iter().map(|e| self.typeck_results().expr_ty_adjusted(e));
                    let tupled_args = Expr {
                        ty: tcx.mk_tup(arg_tys),
                        temp_lifetime,
                        span: expr.span,
                        kind: ExprKind::Tuple { fields: self.mirror_exprs(args) },
                    };
                    let tupled_args = self.thir.exprs.push(tupled_args);

                    ExprKind::Call {
                        ty: method.ty,
                        fun: self.thir.exprs.push(method),
                        args: Box::new([self.mirror_expr(fun), tupled_args]),
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
                        let substs = self.typeck_results().node_substs(fun.hir_id);
                        let user_provided_types = self.typeck_results().user_provided_types();
                        let user_ty =
                            user_provided_types.get(fun.hir_id).copied().map(|mut u_ty| {
                                if let UserType::TypeOf(ref mut did, _) = &mut u_ty.value {
                                    *did = adt_def.did();
                                }
                                Box::new(u_ty)
                            });
                        debug!("make_mirror_unadjusted: (call) user_ty={:?}", user_ty);

                        let field_refs = args
                            .iter()
                            .enumerate()
                            .map(|(idx, e)| FieldExpr {
                                name: Field::new(idx),
                                expr: self.mirror_expr(e),
                            })
                            .collect();
                        ExprKind::Adt(Box::new(AdtExpr {
                            adt_def,
                            substs,
                            variant_index: index,
                            fields: field_refs,
                            user_ty,
                            base: None,
                        }))
                    } else {
                        ExprKind::Call {
                            ty: self.typeck_results().node_type(fun.hir_id),
                            fun: self.mirror_expr(fun),
                            args: self.mirror_exprs(args),
                            from_hir_call: true,
                            fn_span: expr.span,
                        }
                    }
                }
            }

            hir::ExprKind::AddrOf(hir::BorrowKind::Ref, mutbl, ref arg) => {
                ExprKind::Borrow { borrow_kind: mutbl.to_borrow_kind(), arg: self.mirror_expr(arg) }
            }

            hir::ExprKind::AddrOf(hir::BorrowKind::Raw, mutability, ref arg) => {
                ExprKind::AddressOf { mutability, arg: self.mirror_expr(arg) }
            }

            hir::ExprKind::Block(ref blk, _) => ExprKind::Block { block: self.mirror_block(blk) },

            hir::ExprKind::Assign(ref lhs, ref rhs, _) => {
                ExprKind::Assign { lhs: self.mirror_expr(lhs), rhs: self.mirror_expr(rhs) }
            }

            hir::ExprKind::AssignOp(op, ref lhs, ref rhs) => {
                if self.typeck_results().is_method_call(expr) {
                    let lhs = self.mirror_expr(lhs);
                    let rhs = self.mirror_expr(rhs);
                    self.overloaded_operator(expr, Box::new([lhs, rhs]))
                } else {
                    ExprKind::AssignOp {
                        op: bin_op(op.node),
                        lhs: self.mirror_expr(lhs),
                        rhs: self.mirror_expr(rhs),
                    }
                }
            }

            hir::ExprKind::Lit(ref lit) => ExprKind::Literal { lit, neg: false },

            hir::ExprKind::Binary(op, ref lhs, ref rhs) => {
                if self.typeck_results().is_method_call(expr) {
                    let lhs = self.mirror_expr(lhs);
                    let rhs = self.mirror_expr(rhs);
                    self.overloaded_operator(expr, Box::new([lhs, rhs]))
                } else {
                    // FIXME overflow
                    match op.node {
                        hir::BinOpKind::And => ExprKind::LogicalOp {
                            op: LogicalOp::And,
                            lhs: self.mirror_expr(lhs),
                            rhs: self.mirror_expr(rhs),
                        },
                        hir::BinOpKind::Or => ExprKind::LogicalOp {
                            op: LogicalOp::Or,
                            lhs: self.mirror_expr(lhs),
                            rhs: self.mirror_expr(rhs),
                        },
                        _ => {
                            let op = bin_op(op.node);
                            ExprKind::Binary {
                                op,
                                lhs: self.mirror_expr(lhs),
                                rhs: self.mirror_expr(rhs),
                            }
                        }
                    }
                }
            }

            hir::ExprKind::Index(ref lhs, ref index) => {
                if self.typeck_results().is_method_call(expr) {
                    let lhs = self.mirror_expr(lhs);
                    let index = self.mirror_expr(index);
                    self.overloaded_place(expr, expr_ty, None, Box::new([lhs, index]), expr.span)
                } else {
                    ExprKind::Index { lhs: self.mirror_expr(lhs), index: self.mirror_expr(index) }
                }
            }

            hir::ExprKind::Unary(hir::UnOp::Deref, ref arg) => {
                if self.typeck_results().is_method_call(expr) {
                    let arg = self.mirror_expr(arg);
                    self.overloaded_place(expr, expr_ty, None, Box::new([arg]), expr.span)
                } else {
                    ExprKind::Deref { arg: self.mirror_expr(arg) }
                }
            }

            hir::ExprKind::Unary(hir::UnOp::Not, ref arg) => {
                if self.typeck_results().is_method_call(expr) {
                    let arg = self.mirror_expr(arg);
                    self.overloaded_operator(expr, Box::new([arg]))
                } else {
                    ExprKind::Unary { op: UnOp::Not, arg: self.mirror_expr(arg) }
                }
            }

            hir::ExprKind::Unary(hir::UnOp::Neg, ref arg) => {
                if self.typeck_results().is_method_call(expr) {
                    let arg = self.mirror_expr(arg);
                    self.overloaded_operator(expr, Box::new([arg]))
                } else if let hir::ExprKind::Lit(ref lit) = arg.kind {
                    ExprKind::Literal { lit, neg: true }
                } else {
                    ExprKind::Unary { op: UnOp::Neg, arg: self.mirror_expr(arg) }
                }
            }

            hir::ExprKind::Struct(ref qpath, ref fields, ref base) => match expr_ty.kind() {
                ty::Adt(adt, substs) => match adt.adt_kind() {
                    AdtKind::Struct | AdtKind::Union => {
                        let user_provided_types = self.typeck_results().user_provided_types();
                        let user_ty = user_provided_types.get(expr.hir_id).copied().map(Box::new);
                        debug!("make_mirror_unadjusted: (struct/union) user_ty={:?}", user_ty);
                        ExprKind::Adt(Box::new(AdtExpr {
                            adt_def: *adt,
                            variant_index: VariantIdx::new(0),
                            substs,
                            user_ty,
                            fields: self.field_refs(fields),
                            base: base.as_ref().map(|base| FruInfo {
                                base: self.mirror_expr(base),
                                field_types: self.typeck_results().fru_field_types()[expr.hir_id]
                                    .iter()
                                    .copied()
                                    .collect(),
                            }),
                        }))
                    }
                    AdtKind::Enum => {
                        let res = self.typeck_results().qpath_res(qpath, expr.hir_id);
                        match res {
                            Res::Def(DefKind::Variant, variant_id) => {
                                assert!(base.is_none());

                                let index = adt.variant_index_with_id(variant_id);
                                let user_provided_types =
                                    self.typeck_results().user_provided_types();
                                let user_ty =
                                    user_provided_types.get(expr.hir_id).copied().map(Box::new);
                                debug!("make_mirror_unadjusted: (variant) user_ty={:?}", user_ty);
                                ExprKind::Adt(Box::new(AdtExpr {
                                    adt_def: *adt,
                                    variant_index: index,
                                    substs,
                                    user_ty,
                                    fields: self.field_refs(fields),
                                    base: None,
                                }))
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

            hir::ExprKind::Closure { .. } => {
                let closure_ty = self.typeck_results().expr_ty(expr);
                let (def_id, substs, movability) = match *closure_ty.kind() {
                    ty::Closure(def_id, substs) => (def_id, UpvarSubsts::Closure(substs), None),
                    ty::Generator(def_id, substs, movability) => {
                        (def_id, UpvarSubsts::Generator(substs), Some(movability))
                    }
                    _ => {
                        span_bug!(expr.span, "closure expr w/o closure type: {:?}", closure_ty);
                    }
                };
                let def_id = def_id.expect_local();

                let upvars = self
                    .typeck_results
                    .closure_min_captures_flattened(def_id)
                    .zip(substs.upvar_tys())
                    .map(|(captured_place, ty)| {
                        let upvars = self.capture_upvar(expr, captured_place, ty);
                        self.thir.exprs.push(upvars)
                    })
                    .collect();

                // Convert the closure fake reads, if any, from hir `Place` to ExprRef
                let fake_reads = match self.typeck_results.closure_fake_reads.get(&def_id) {
                    Some(fake_reads) => fake_reads
                        .iter()
                        .map(|(place, cause, hir_id)| {
                            let expr = self.convert_captured_hir_place(expr, place.clone());
                            (self.thir.exprs.push(expr), *cause, *hir_id)
                        })
                        .collect(),
                    None => Vec::new(),
                };

                ExprKind::Closure(Box::new(ClosureExpr {
                    closure_id: def_id,
                    substs,
                    upvars,
                    movability,
                    fake_reads,
                }))
            }

            hir::ExprKind::Path(ref qpath) => {
                let res = self.typeck_results().qpath_res(qpath, expr.hir_id);
                self.convert_path_expr(expr, res)
            }

            hir::ExprKind::InlineAsm(ref asm) => ExprKind::InlineAsm(Box::new(InlineAsmExpr {
                template: asm.template,
                operands: asm
                    .operands
                    .iter()
                    .map(|(op, _op_sp)| match *op {
                        hir::InlineAsmOperand::In { reg, ref expr } => {
                            InlineAsmOperand::In { reg, expr: self.mirror_expr(expr) }
                        }
                        hir::InlineAsmOperand::Out { reg, late, ref expr } => {
                            InlineAsmOperand::Out {
                                reg,
                                late,
                                expr: expr.as_ref().map(|expr| self.mirror_expr(expr)),
                            }
                        }
                        hir::InlineAsmOperand::InOut { reg, late, ref expr } => {
                            InlineAsmOperand::InOut { reg, late, expr: self.mirror_expr(expr) }
                        }
                        hir::InlineAsmOperand::SplitInOut {
                            reg,
                            late,
                            ref in_expr,
                            ref out_expr,
                        } => InlineAsmOperand::SplitInOut {
                            reg,
                            late,
                            in_expr: self.mirror_expr(in_expr),
                            out_expr: out_expr.as_ref().map(|expr| self.mirror_expr(expr)),
                        },
                        hir::InlineAsmOperand::Const { ref anon_const } => {
                            let anon_const_def_id = tcx.hir().local_def_id(anon_const.hir_id);
                            let value = mir::ConstantKind::from_anon_const(
                                tcx,
                                anon_const_def_id,
                                self.param_env,
                            );
                            let span = tcx.hir().span(anon_const.hir_id);

                            InlineAsmOperand::Const { value, span }
                        }
                        hir::InlineAsmOperand::SymFn { ref anon_const } => {
                            let anon_const_def_id = tcx.hir().local_def_id(anon_const.hir_id);
                            let value = mir::ConstantKind::from_anon_const(
                                tcx,
                                anon_const_def_id,
                                self.param_env,
                            );
                            let span = tcx.hir().span(anon_const.hir_id);

                            InlineAsmOperand::SymFn { value, span }
                        }
                        hir::InlineAsmOperand::SymStatic { path: _, def_id } => {
                            InlineAsmOperand::SymStatic { def_id }
                        }
                    })
                    .collect(),
                options: asm.options,
                line_spans: asm.line_spans,
            })),

            hir::ExprKind::ConstBlock(ref anon_const) => {
                let ty = self.typeck_results().node_type(anon_const.hir_id);
                let did = tcx.hir().local_def_id(anon_const.hir_id).to_def_id();
                let typeck_root_def_id = tcx.typeck_root_def_id(did);
                let parent_substs =
                    tcx.erase_regions(InternalSubsts::identity_for_item(tcx, typeck_root_def_id));
                let substs =
                    InlineConstSubsts::new(tcx, InlineConstSubstsParts { parent_substs, ty })
                        .substs;

                ExprKind::ConstBlock { did, substs }
            }
            // Now comes the rote stuff:
            hir::ExprKind::Repeat(ref v, _) => {
                let ty = self.typeck_results().expr_ty(expr);
                let ty::Array(_, count) = ty.kind() else {
                    span_bug!(expr.span, "unexpected repeat expr ty: {:?}", ty);
                };

                ExprKind::Repeat { value: self.mirror_expr(v), count: *count }
            }
            hir::ExprKind::Ret(ref v) => {
                ExprKind::Return { value: v.as_ref().map(|v| self.mirror_expr(v)) }
            }
            hir::ExprKind::Break(dest, ref value) => match dest.target_id {
                Ok(target_id) => ExprKind::Break {
                    label: region::Scope { id: target_id.local_id, data: region::ScopeData::Node },
                    value: value.as_ref().map(|value| self.mirror_expr(value)),
                },
                Err(err) => bug!("invalid loop id for break: {}", err),
            },
            hir::ExprKind::Continue(dest) => match dest.target_id {
                Ok(loop_id) => ExprKind::Continue {
                    label: region::Scope { id: loop_id.local_id, data: region::ScopeData::Node },
                },
                Err(err) => bug!("invalid loop id for continue: {}", err),
            },
            hir::ExprKind::Let(let_expr) => ExprKind::Let {
                expr: self.mirror_expr(let_expr.init),
                pat: self.pattern_from_hir(let_expr.pat),
            },
            hir::ExprKind::If(cond, then, else_opt) => ExprKind::If {
                if_then_scope: region::Scope {
                    id: then.hir_id.local_id,
                    data: region::ScopeData::IfThen,
                },
                cond: self.mirror_expr(cond),
                then: self.mirror_expr(then),
                else_opt: else_opt.map(|el| self.mirror_expr(el)),
            },
            hir::ExprKind::Match(ref discr, ref arms, _) => ExprKind::Match {
                scrutinee: self.mirror_expr(discr),
                arms: arms.iter().map(|a| self.convert_arm(a)).collect(),
            },
            hir::ExprKind::Loop(ref body, ..) => {
                let block_ty = self.typeck_results().node_type(body.hir_id);
                let temp_lifetime = self
                    .rvalue_scopes
                    .temporary_scope(self.region_scope_tree, body.hir_id.local_id);
                let block = self.mirror_block(body);
                let body = self.thir.exprs.push(Expr {
                    ty: block_ty,
                    temp_lifetime,
                    span: self.thir[block].span,
                    kind: ExprKind::Block { block },
                });
                ExprKind::Loop { body }
            }
            hir::ExprKind::Field(ref source, ..) => ExprKind::Field {
                lhs: self.mirror_expr(source),
                variant_index: VariantIdx::new(0),
                name: Field::new(tcx.field_index(expr.hir_id, self.typeck_results)),
            },
            hir::ExprKind::Cast(ref source, ref cast_ty) => {
                // Check for a user-given type annotation on this `cast`
                let user_provided_types = self.typeck_results.user_provided_types();
                let user_ty = user_provided_types.get(cast_ty.hir_id);

                debug!(
                    "cast({:?}) has ty w/ hir_id {:?} and user provided ty {:?}",
                    expr, cast_ty.hir_id, user_ty,
                );

                let cast = self.mirror_expr_cast(*source, temp_lifetime, expr.span);

                if let Some(user_ty) = user_ty {
                    // NOTE: Creating a new Expr and wrapping a Cast inside of it may be
                    //       inefficient, revisit this when performance becomes an issue.
                    let cast_expr = self.thir.exprs.push(Expr {
                        temp_lifetime,
                        ty: expr_ty,
                        span: expr.span,
                        kind: cast,
                    });
                    debug!("make_mirror_unadjusted: (cast) user_ty={:?}", user_ty);

                    ExprKind::ValueTypeAscription {
                        source: cast_expr,
                        user_ty: Some(Box::new(*user_ty)),
                    }
                } else {
                    cast
                }
            }
            hir::ExprKind::Type(ref source, ref ty) => {
                let user_provided_types = self.typeck_results.user_provided_types();
                let user_ty = user_provided_types.get(ty.hir_id).copied().map(Box::new);
                debug!("make_mirror_unadjusted: (type) user_ty={:?}", user_ty);
                let mirrored = self.mirror_expr(source);
                if source.is_syntactic_place_expr() {
                    ExprKind::PlaceTypeAscription { source: mirrored, user_ty }
                } else {
                    ExprKind::ValueTypeAscription { source: mirrored, user_ty }
                }
            }
            hir::ExprKind::DropTemps(ref source) => {
                ExprKind::Use { source: self.mirror_expr(source) }
            }
            hir::ExprKind::Box(ref value) => ExprKind::Box { value: self.mirror_expr(value) },
            hir::ExprKind::Array(ref fields) => {
                ExprKind::Array { fields: self.mirror_exprs(fields) }
            }
            hir::ExprKind::Tup(ref fields) => ExprKind::Tuple { fields: self.mirror_exprs(fields) },

            hir::ExprKind::Yield(ref v, _) => ExprKind::Yield { value: self.mirror_expr(v) },
            hir::ExprKind::Err => unreachable!(),
        };

        Expr { temp_lifetime, ty: expr_ty, span: expr.span, kind }
    }

    fn user_substs_applied_to_res(
        &mut self,
        hir_id: hir::HirId,
        res: Res,
    ) -> Option<Box<ty::CanonicalUserType<'tcx>>> {
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
                self.typeck_results().user_provided_types().get(hir_id).copied().map(Box::new)
            }

            // A unit struct/variant which is used as a value (e.g.,
            // `None`). This has the type of the enum/struct that defines
            // this variant -- but with the substitutions given by the
            // user.
            Res::Def(DefKind::Ctor(_, CtorKind::Const), _) => {
                self.user_substs_applied_to_ty_of_hir_id(hir_id).map(Box::new)
            }

            // `Self` is used in expression as a tuple struct constructor or a unit struct constructor
            Res::SelfCtor(_) => self.user_substs_applied_to_ty_of_hir_id(hir_id).map(Box::new),

            _ => bug!("user_substs_applied_to_res: unexpected res {:?} at {:?}", res, hir_id),
        };
        debug!("user_substs_applied_to_res: user_provided_type={:?}", user_provided_type);
        user_provided_type
    }

    fn method_callee(
        &mut self,
        expr: &hir::Expr<'_>,
        span: Span,
        overloaded_callee: Option<(DefId, SubstsRef<'tcx>)>,
    ) -> Expr<'tcx> {
        let temp_lifetime =
            self.rvalue_scopes.temporary_scope(self.region_scope_tree, expr.hir_id.local_id);
        let (def_id, substs, user_ty) = match overloaded_callee {
            Some((def_id, substs)) => (def_id, substs, None),
            None => {
                let (kind, def_id) =
                    self.typeck_results().type_dependent_def(expr.hir_id).unwrap_or_else(|| {
                        span_bug!(expr.span, "no type-dependent def for method callee")
                    });
                let user_ty = self.user_substs_applied_to_res(expr.hir_id, Res::Def(kind, def_id));
                debug!("method_callee: user_ty={:?}", user_ty);
                (def_id, self.typeck_results().node_substs(expr.hir_id), user_ty)
            }
        };
        let ty = self.tcx().mk_fn_def(def_id, substs);
        Expr { temp_lifetime, ty, span, kind: ExprKind::ZstLiteral { user_ty } }
    }

    fn convert_arm(&mut self, arm: &'tcx hir::Arm<'tcx>) -> ArmId {
        let arm = Arm {
            pattern: self.pattern_from_hir(&arm.pat),
            guard: arm.guard.as_ref().map(|g| match g {
                hir::Guard::If(ref e) => Guard::If(self.mirror_expr(e)),
                hir::Guard::IfLet(ref l) => {
                    Guard::IfLet(self.pattern_from_hir(l.pat), self.mirror_expr(l.init))
                }
            }),
            body: self.mirror_expr(arm.body),
            lint_level: LintLevel::Explicit(arm.hir_id),
            scope: region::Scope { id: arm.hir_id.local_id, data: region::ScopeData::Node },
            span: arm.span,
        };
        self.thir.arms.push(arm)
    }

    fn convert_path_expr(&mut self, expr: &'tcx hir::Expr<'tcx>, res: Res) -> ExprKind<'tcx> {
        let substs = self.typeck_results().node_substs(expr.hir_id);
        match res {
            // A regular function, constructor function or a constant.
            Res::Def(DefKind::Fn, _)
            | Res::Def(DefKind::AssocFn, _)
            | Res::Def(DefKind::Ctor(_, CtorKind::Fn), _)
            | Res::SelfCtor(_) => {
                let user_ty = self.user_substs_applied_to_res(expr.hir_id, res);
                ExprKind::ZstLiteral { user_ty }
            }

            Res::Def(DefKind::ConstParam, def_id) => {
                let hir_id = self.tcx.hir().local_def_id_to_hir_id(def_id.expect_local());
                let item_id = self.tcx.hir().get_parent_node(hir_id);
                let item_def_id = self.tcx.hir().local_def_id(item_id);
                let generics = self.tcx.generics_of(item_def_id);
                let index = generics.param_def_id_to_index[&def_id];
                let name = self.tcx.hir().name(hir_id);
                let param = ty::ParamConst::new(index, name);

                ExprKind::ConstParam { param, def_id }
            }

            Res::Def(DefKind::Const, def_id) | Res::Def(DefKind::AssocConst, def_id) => {
                let user_ty = self.user_substs_applied_to_res(expr.hir_id, res);
                ExprKind::NamedConst { def_id, substs, user_ty }
            }

            Res::Def(DefKind::Ctor(_, CtorKind::Const), def_id) => {
                let user_provided_types = self.typeck_results.user_provided_types();
                let user_ty = user_provided_types.get(expr.hir_id).copied().map(Box::new);
                debug!("convert_path_expr: user_ty={:?}", user_ty);
                let ty = self.typeck_results().node_type(expr.hir_id);
                match ty.kind() {
                    // A unit struct/variant which is used as a value.
                    // We return a completely different ExprKind here to account for this special case.
                    ty::Adt(adt_def, substs) => ExprKind::Adt(Box::new(AdtExpr {
                        adt_def: *adt_def,
                        variant_index: adt_def.variant_index_with_ctor_id(def_id),
                        substs,
                        user_ty,
                        fields: Box::new([]),
                        base: None,
                    })),
                    _ => bug!("unexpected ty: {:?}", ty),
                }
            }

            // We encode uses of statics as a `*&STATIC` where the `&STATIC` part is
            // a constant reference (or constant raw pointer for `static mut`) in MIR
            Res::Def(DefKind::Static(_), id) => {
                let ty = self.tcx.static_ptr_ty(id);
                let temp_lifetime = self
                    .rvalue_scopes
                    .temporary_scope(self.region_scope_tree, expr.hir_id.local_id);
                let kind = if self.tcx.is_thread_local_static(id) {
                    ExprKind::ThreadLocalRef(id)
                } else {
                    let alloc_id = self.tcx.create_static_alloc(id);
                    ExprKind::StaticRef { alloc_id, ty, def_id: id }
                };
                ExprKind::Deref {
                    arg: self.thir.exprs.push(Expr { ty, temp_lifetime, span: expr.span, kind }),
                }
            }

            Res::Local(var_hir_id) => self.convert_var(var_hir_id),

            _ => span_bug!(expr.span, "res `{:?}` not yet implemented", res),
        }
    }

    fn convert_var(&mut self, var_hir_id: hir::HirId) -> ExprKind<'tcx> {
        // We want upvars here not captures.
        // Captures will be handled in MIR.
        let is_upvar = self
            .tcx
            .upvars_mentioned(self.body_owner)
            .map_or(false, |upvars| upvars.contains_key(&var_hir_id));

        debug!(
            "convert_var({:?}): is_upvar={}, body_owner={:?}",
            var_hir_id, is_upvar, self.body_owner
        );

        if is_upvar {
            ExprKind::UpvarRef {
                closure_def_id: self.body_owner,
                var_hir_id: LocalVarId(var_hir_id),
            }
        } else {
            ExprKind::VarRef { id: LocalVarId(var_hir_id) }
        }
    }

    fn overloaded_operator(
        &mut self,
        expr: &'tcx hir::Expr<'tcx>,
        args: Box<[ExprId]>,
    ) -> ExprKind<'tcx> {
        let fun = self.method_callee(expr, expr.span, None);
        let fun = self.thir.exprs.push(fun);
        ExprKind::Call {
            ty: self.thir[fun].ty,
            fun,
            args,
            from_hir_call: false,
            fn_span: expr.span,
        }
    }

    fn overloaded_place(
        &mut self,
        expr: &'tcx hir::Expr<'tcx>,
        place_ty: Ty<'tcx>,
        overloaded_callee: Option<(DefId, SubstsRef<'tcx>)>,
        args: Box<[ExprId]>,
        span: Span,
    ) -> ExprKind<'tcx> {
        // For an overloaded *x or x[y] expression of type T, the method
        // call returns an &T and we must add the deref so that the types
        // line up (this is because `*x` and `x[y]` represent places):

        // Reconstruct the output assuming it's a reference with the
        // same region and mutability as the receiver. This holds for
        // `Deref(Mut)::Deref(_mut)` and `Index(Mut)::index(_mut)`.
        let ty::Ref(region, _, mutbl) = *self.thir[args[0]].ty.kind() else {
            span_bug!(span, "overloaded_place: receiver is not a reference");
        };
        let ref_ty = self.tcx.mk_ref(region, ty::TypeAndMut { ty: place_ty, mutbl });

        // construct the complete expression `foo()` for the overloaded call,
        // which will yield the &T type
        let temp_lifetime =
            self.rvalue_scopes.temporary_scope(self.region_scope_tree, expr.hir_id.local_id);
        let fun = self.method_callee(expr, span, overloaded_callee);
        let fun = self.thir.exprs.push(fun);
        let fun_ty = self.thir[fun].ty;
        let ref_expr = self.thir.exprs.push(Expr {
            temp_lifetime,
            ty: ref_ty,
            span,
            kind: ExprKind::Call { ty: fun_ty, fun, args, from_hir_call: false, fn_span: span },
        });

        // construct and return a deref wrapper `*foo()`
        ExprKind::Deref { arg: ref_expr }
    }

    fn convert_captured_hir_place(
        &mut self,
        closure_expr: &'tcx hir::Expr<'tcx>,
        place: HirPlace<'tcx>,
    ) -> Expr<'tcx> {
        let temp_lifetime = self
            .rvalue_scopes
            .temporary_scope(self.region_scope_tree, closure_expr.hir_id.local_id);
        let var_ty = place.base_ty;

        // The result of capture analysis in `rustc_typeck/check/upvar.rs`represents a captured path
        // as it's seen for use within the closure and not at the time of closure creation.
        //
        // That is we see expect to see it start from a captured upvar and not something that is local
        // to the closure's parent.
        let var_hir_id = match place.base {
            HirPlaceBase::Upvar(upvar_id) => upvar_id.var_path.hir_id,
            base => bug!("Expected an upvar, found {:?}", base),
        };

        let mut captured_place_expr = Expr {
            temp_lifetime,
            ty: var_ty,
            span: closure_expr.span,
            kind: self.convert_var(var_hir_id),
        };

        for proj in place.projections.iter() {
            let kind = match proj.kind {
                HirProjectionKind::Deref => {
                    ExprKind::Deref { arg: self.thir.exprs.push(captured_place_expr) }
                }
                HirProjectionKind::Field(field, variant_index) => ExprKind::Field {
                    lhs: self.thir.exprs.push(captured_place_expr),
                    variant_index,
                    name: Field::new(field as usize),
                },
                HirProjectionKind::Index | HirProjectionKind::Subslice => {
                    // We don't capture these projections, so we can ignore them here
                    continue;
                }
            };

            captured_place_expr =
                Expr { temp_lifetime, ty: proj.ty, span: closure_expr.span, kind };
        }

        captured_place_expr
    }

    fn capture_upvar(
        &mut self,
        closure_expr: &'tcx hir::Expr<'tcx>,
        captured_place: &'tcx ty::CapturedPlace<'tcx>,
        upvar_ty: Ty<'tcx>,
    ) -> Expr<'tcx> {
        let upvar_capture = captured_place.info.capture_kind;
        let captured_place_expr =
            self.convert_captured_hir_place(closure_expr, captured_place.place.clone());
        let temp_lifetime = self
            .rvalue_scopes
            .temporary_scope(self.region_scope_tree, closure_expr.hir_id.local_id);

        match upvar_capture {
            ty::UpvarCapture::ByValue => captured_place_expr,
            ty::UpvarCapture::ByRef(upvar_borrow) => {
                let borrow_kind = match upvar_borrow {
                    ty::BorrowKind::ImmBorrow => BorrowKind::Shared,
                    ty::BorrowKind::UniqueImmBorrow => BorrowKind::Unique,
                    ty::BorrowKind::MutBorrow => BorrowKind::Mut { allow_two_phase_borrow: false },
                };
                Expr {
                    temp_lifetime,
                    ty: upvar_ty,
                    span: closure_expr.span,
                    kind: ExprKind::Borrow {
                        borrow_kind,
                        arg: self.thir.exprs.push(captured_place_expr),
                    },
                }
            }
        }
    }

    /// Converts a list of named fields (i.e., for struct-like struct/enum ADTs) into FieldExpr.
    fn field_refs(&mut self, fields: &'tcx [hir::ExprField<'tcx>]) -> Box<[FieldExpr]> {
        fields
            .iter()
            .map(|field| FieldExpr {
                name: Field::new(self.tcx.field_index(field.hir_id, self.typeck_results)),
                expr: self.mirror_expr(field.expr),
            })
            .collect()
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
