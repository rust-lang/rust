use itertools::Itertools;
use rustc_abi::{FIRST_VARIANT, FieldIdx};
use rustc_ast::UnsafeBinderCastKind;
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, CtorOf, DefKind, Res};
use rustc_index::Idx;
use rustc_middle::hir::place::{
    Place as HirPlace, PlaceBase as HirPlaceBase, ProjectionKind as HirProjectionKind,
};
use rustc_middle::middle::region;
use rustc_middle::mir::{self, AssignOp, BinOp, BorrowKind, UnOp};
use rustc_middle::thir::*;
use rustc_middle::ty::adjustment::{
    Adjust, Adjustment, AutoBorrow, AutoBorrowMutability, PointerCoercion,
};
use rustc_middle::ty::{
    self, AdtKind, GenericArgs, InlineConstArgs, InlineConstArgsParts, ScalarInt, Ty, UpvarArgs,
};
use rustc_middle::{bug, span_bug};
use rustc_span::{Span, sym};
use tracing::{debug, info, instrument, trace};

use crate::thir::cx::ThirBuildCx;

impl<'tcx> ThirBuildCx<'tcx> {
    /// Create a THIR expression for the given HIR expression. This expands all
    /// adjustments and directly adds the type information from the
    /// `typeck_results`. See the [dev-guide] for more details.
    ///
    /// (The term "mirror" in this case does not refer to "flipped" or
    /// "reversed".)
    ///
    /// [dev-guide]: https://rustc-dev-guide.rust-lang.org/thir.html
    pub(crate) fn mirror_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) -> ExprId {
        // `mirror_expr` is recursing very deep. Make sure the stack doesn't overflow.
        ensure_sufficient_stack(|| self.mirror_expr_inner(expr))
    }

    pub(crate) fn mirror_exprs(&mut self, exprs: &'tcx [hir::Expr<'tcx>]) -> Box<[ExprId]> {
        exprs.iter().map(|expr| self.mirror_expr_inner(expr)).collect()
    }

    #[instrument(level = "trace", skip(self, hir_expr))]
    pub(super) fn mirror_expr_inner(&mut self, hir_expr: &'tcx hir::Expr<'tcx>) -> ExprId {
        let expr_scope =
            region::Scope { local_id: hir_expr.hir_id.local_id, data: region::ScopeData::Node };

        trace!(?hir_expr.hir_id, ?hir_expr.span);

        let mut expr = self.make_mirror_unadjusted(hir_expr);

        trace!(?expr.ty);

        // Now apply adjustments, if any.
        if self.apply_adjustments {
            for adjustment in self.typeck_results.expr_adjustments(hir_expr) {
                trace!(?expr, ?adjustment);
                let span = expr.span;
                expr = self.apply_adjustment(hir_expr, expr, adjustment, span);
            }
        }

        trace!(?expr.ty, "after adjustments");

        // Finally, wrap this up in the expr's scope.
        expr = Expr {
            temp_lifetime: expr.temp_lifetime,
            ty: expr.ty,
            span: hir_expr.span,
            kind: ExprKind::Scope {
                region_scope: expr_scope,
                value: self.thir.exprs.push(expr),
                lint_level: LintLevel::Explicit(hir_expr.hir_id),
            },
        };

        // OK, all done!
        self.thir.exprs.push(expr)
    }

    #[instrument(level = "trace", skip(self, expr, span))]
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
            Adjust::Pointer(cast) => {
                if cast == PointerCoercion::Unsize {
                    adjust_span(&mut expr);
                }

                let is_from_as_cast = if let hir::Node::Expr(hir::Expr {
                    kind: hir::ExprKind::Cast(..),
                    span: cast_span,
                    ..
                }) = self.tcx.parent_hir_node(hir_expr.hir_id)
                {
                    // Use the whole span of the `x as T` expression for the coercion.
                    span = *cast_span;
                    true
                } else {
                    false
                };
                ExprKind::PointerCoercion {
                    cast,
                    source: self.thir.exprs.push(expr),
                    is_from_as_cast,
                }
            }
            Adjust::NeverToAny if adjustment.target.is_never() => return expr,
            Adjust::NeverToAny => ExprKind::NeverToAny { source: self.thir.exprs.push(expr) },
            Adjust::Deref(None) => {
                adjust_span(&mut expr);
                ExprKind::Deref { arg: self.thir.exprs.push(expr) }
            }
            Adjust::Deref(Some(deref)) => {
                // We don't need to do call adjust_span here since
                // deref coercions always start with a built-in deref.
                let call_def_id = deref.method_call(self.tcx);
                let overloaded_callee =
                    Ty::new_fn_def(self.tcx, call_def_id, self.tcx.mk_args(&[expr.ty.into()]));

                expr = Expr {
                    temp_lifetime,
                    ty: Ty::new_ref(self.tcx, self.tcx.lifetimes.re_erased, expr.ty, deref.mutbl),
                    span,
                    kind: ExprKind::Borrow {
                        borrow_kind: deref.mutbl.to_borrow_kind(),
                        arg: self.thir.exprs.push(expr),
                    },
                };

                let expr = Box::new([self.thir.exprs.push(expr)]);

                self.overloaded_place(
                    hir_expr,
                    adjustment.target,
                    Some(overloaded_callee),
                    expr,
                    deref.span,
                )
            }
            Adjust::Borrow(AutoBorrow::Ref(m)) => ExprKind::Borrow {
                borrow_kind: m.to_borrow_kind(),
                arg: self.thir.exprs.push(expr),
            },
            Adjust::Borrow(AutoBorrow::RawPtr(mutability)) => {
                ExprKind::RawBorrow { mutability, arg: self.thir.exprs.push(expr) }
            }
            Adjust::ReborrowPin(mutbl) => {
                debug!("apply ReborrowPin adjustment");
                // Rewrite `$expr` as `Pin { __pointer: &(mut)? *($expr).__pointer }`

                // We'll need these types later on
                let pin_ty_args = match expr.ty.kind() {
                    ty::Adt(_, args) => args,
                    _ => bug!("ReborrowPin with non-Pin type"),
                };
                let pin_ty = pin_ty_args.iter().next().unwrap().expect_ty();
                let ptr_target_ty = match pin_ty.kind() {
                    ty::Ref(_, ty, _) => *ty,
                    _ => bug!("ReborrowPin with non-Ref type"),
                };

                // pointer = ($expr).__pointer
                let pointer_target = ExprKind::Field {
                    lhs: self.thir.exprs.push(expr),
                    variant_index: FIRST_VARIANT,
                    name: FieldIdx::ZERO,
                };
                let arg = Expr { temp_lifetime, ty: pin_ty, span, kind: pointer_target };
                let arg = self.thir.exprs.push(arg);

                // arg = *pointer
                let expr = ExprKind::Deref { arg };
                let arg = self.thir.exprs.push(Expr {
                    temp_lifetime,
                    ty: ptr_target_ty,
                    span,
                    kind: expr,
                });

                // expr = &mut target
                let borrow_kind = match mutbl {
                    hir::Mutability::Mut => BorrowKind::Mut { kind: mir::MutBorrowKind::Default },
                    hir::Mutability::Not => BorrowKind::Shared,
                };
                let new_pin_target =
                    Ty::new_ref(self.tcx, self.tcx.lifetimes.re_erased, ptr_target_ty, mutbl);
                let expr = self.thir.exprs.push(Expr {
                    temp_lifetime,
                    ty: new_pin_target,
                    span,
                    kind: ExprKind::Borrow { borrow_kind, arg },
                });

                // kind = Pin { __pointer: pointer }
                let pin_did = self.tcx.require_lang_item(rustc_hir::LangItem::Pin, Some(span));
                let args = self.tcx.mk_args(&[new_pin_target.into()]);
                let kind = ExprKind::Adt(Box::new(AdtExpr {
                    adt_def: self.tcx.adt_def(pin_did),
                    variant_index: FIRST_VARIANT,
                    args,
                    fields: Box::new([FieldExpr { name: FieldIdx::ZERO, expr }]),
                    user_ty: None,
                    base: AdtExprBase::None,
                }));

                debug!(?kind);
                kind
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
        temp_lifetime: TempLifetime,
        span: Span,
    ) -> ExprKind<'tcx> {
        let tcx = self.tcx;

        // Check to see if this cast is a "coercion cast", where the cast is actually done
        // using a coercion (or is a no-op).
        if self.typeck_results.is_coercion_cast(source.hir_id) {
            // Convert the lexpr to a vexpr.
            ExprKind::Use { source: self.mirror_expr(source) }
        } else if self.typeck_results.expr_ty(source).is_ref() {
            // Special cased so that we can type check that the element
            // type of the source matches the pointed to type of the
            // destination.
            ExprKind::PointerCoercion {
                source: self.mirror_expr(source),
                cast: PointerCoercion::ArrayToPointer,
                is_from_as_cast: true,
            }
        } else if let hir::ExprKind::Path(ref qpath) = source.kind
            && let res = self.typeck_results.qpath_res(qpath, source.hir_id)
            && let ty = self.typeck_results.node_type(source.hir_id)
            && let ty::Adt(adt_def, args) = ty.kind()
            && let Res::Def(DefKind::Ctor(CtorOf::Variant, CtorKind::Const), variant_ctor_id) = res
        {
            // Check whether this is casting an enum variant discriminant.
            // To prevent cycles, we refer to the discriminant initializer,
            // which is always an integer and thus doesn't need to know the
            // enum's layout (or its tag type) to compute it during const eval.
            // Example:
            // enum Foo {
            //     A,
            //     B = A as isize + 4,
            // }
            // The correct solution would be to add symbolic computations to miri,
            // so we wouldn't have to compute and store the actual value

            let idx = adt_def.variant_index_with_ctor_id(variant_ctor_id);
            let (discr_did, discr_offset) = adt_def.discriminant_def_for_variant(idx);

            use rustc_middle::ty::util::IntTypeExt;
            let ty = adt_def.repr().discr_type();
            let discr_ty = ty.to_ty(tcx);

            let size = tcx
                .layout_of(self.typing_env.as_query_input(discr_ty))
                .unwrap_or_else(|e| panic!("could not compute layout for {discr_ty:?}: {e:?}"))
                .size;

            let (lit, overflowing) = ScalarInt::truncate_from_uint(discr_offset as u128, size);
            if overflowing {
                // An erroneous enum with too many variants for its repr will emit E0081 and E0370
                self.tcx.dcx().span_delayed_bug(
                    source.span,
                    "overflowing enum wasn't rejected by hir analysis",
                );
            }
            let kind = ExprKind::NonHirLiteral { lit, user_ty: None };
            let offset = self.thir.exprs.push(Expr { temp_lifetime, ty: discr_ty, span, kind });

            let source = match discr_did {
                // in case we are offsetting from a computed discriminant
                // and not the beginning of discriminants (which is always `0`)
                Some(did) => {
                    let kind = ExprKind::NamedConst { def_id: did, args, user_ty: None };
                    let lhs =
                        self.thir.exprs.push(Expr { temp_lifetime, ty: discr_ty, span, kind });
                    let bin = ExprKind::Binary { op: BinOp::Add, lhs, rhs: offset };
                    self.thir.exprs.push(Expr { temp_lifetime, ty: discr_ty, span, kind: bin })
                }
                None => offset,
            };

            ExprKind::Cast { source }
        } else {
            // Default to `ExprKind::Cast` for all explicit casts.
            // MIR building then picks the right MIR casts based on the types.
            ExprKind::Cast { source: self.mirror_expr(source) }
        }
    }

    #[instrument(level = "debug", skip(self), ret)]
    fn make_mirror_unadjusted(&mut self, expr: &'tcx hir::Expr<'tcx>) -> Expr<'tcx> {
        let tcx = self.tcx;
        let expr_ty = self.typeck_results.expr_ty(expr);
        let (temp_lifetime, backwards_incompatible) =
            self.rvalue_scopes.temporary_scope(self.region_scope_tree, expr.hir_id.local_id);

        let kind = match expr.kind {
            // Here comes the interesting stuff:
            hir::ExprKind::MethodCall(segment, receiver, args, fn_span) => {
                // Rewrite a.b(c) into UFCS form like Trait::b(a, c)
                let expr = self.method_callee(expr, segment.ident.span, None);
                info!("Using method span: {:?}", expr.span);
                let args = std::iter::once(receiver)
                    .chain(args.iter())
                    .map(|expr| self.mirror_expr(expr))
                    .collect();
                ExprKind::Call {
                    ty: expr.ty,
                    fun: self.thir.exprs.push(expr),
                    args,
                    from_hir_call: true,
                    fn_span,
                }
            }

            hir::ExprKind::Call(fun, ref args) => {
                if self.typeck_results.is_method_call(expr) {
                    // The callee is something implementing Fn, FnMut, or FnOnce.
                    // Find the actual method implementation being called and
                    // build the appropriate UFCS call expression with the
                    // callee-object as expr parameter.

                    // rewrite f(u, v) into FnOnce::call_once(f, (u, v))

                    let method = self.method_callee(expr, fun.span, None);

                    let arg_tys = args.iter().map(|e| self.typeck_results.expr_ty_adjusted(e));
                    let tupled_args = Expr {
                        ty: Ty::new_tup_from_iter(tcx, arg_tys),
                        temp_lifetime: TempLifetime { temp_lifetime, backwards_incompatible },
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
                } else if let ty::FnDef(def_id, _) = self.typeck_results.expr_ty(fun).kind()
                    && let Some(intrinsic) = self.tcx.intrinsic(def_id)
                    && intrinsic.name == sym::box_new
                {
                    // We don't actually evaluate `fun` here, so make sure that doesn't miss any side-effects.
                    if !matches!(fun.kind, hir::ExprKind::Path(_)) {
                        span_bug!(
                            expr.span,
                            "`box_new` intrinsic can only be called via path expression"
                        );
                    }
                    let value = &args[0];
                    return Expr {
                        temp_lifetime: TempLifetime { temp_lifetime, backwards_incompatible },
                        ty: expr_ty,
                        span: expr.span,
                        kind: ExprKind::Box { value: self.mirror_expr(value) },
                    };
                } else {
                    // Tuple-like ADTs are represented as ExprKind::Call. We convert them here.
                    let adt_data = if let hir::ExprKind::Path(ref qpath) = fun.kind
                        && let Some(adt_def) = expr_ty.ty_adt_def()
                    {
                        match qpath {
                            hir::QPath::Resolved(_, path) => match path.res {
                                Res::Def(DefKind::Ctor(_, CtorKind::Fn), ctor_id) => {
                                    Some((adt_def, adt_def.variant_index_with_ctor_id(ctor_id)))
                                }
                                Res::SelfCtor(..) => Some((adt_def, FIRST_VARIANT)),
                                _ => None,
                            },
                            hir::QPath::TypeRelative(_ty, _) => {
                                if let Some((DefKind::Ctor(_, CtorKind::Fn), ctor_id)) =
                                    self.typeck_results.type_dependent_def(fun.hir_id)
                                {
                                    Some((adt_def, adt_def.variant_index_with_ctor_id(ctor_id)))
                                } else {
                                    None
                                }
                            }
                            _ => None,
                        }
                    } else {
                        None
                    };
                    if let Some((adt_def, index)) = adt_data {
                        let node_args = self.typeck_results.node_args(fun.hir_id);
                        let user_provided_types = self.typeck_results.user_provided_types();
                        let user_ty =
                            user_provided_types.get(fun.hir_id).copied().map(|mut u_ty| {
                                if let ty::UserTypeKind::TypeOf(did, _) = &mut u_ty.value.kind {
                                    *did = adt_def.did();
                                }
                                Box::new(u_ty)
                            });
                        debug!("make_mirror_unadjusted: (call) user_ty={:?}", user_ty);

                        let field_refs = args
                            .iter()
                            .enumerate()
                            .map(|(idx, e)| FieldExpr {
                                name: FieldIdx::new(idx),
                                expr: self.mirror_expr(e),
                            })
                            .collect();
                        ExprKind::Adt(Box::new(AdtExpr {
                            adt_def,
                            args: node_args,
                            variant_index: index,
                            fields: field_refs,
                            user_ty,
                            base: AdtExprBase::None,
                        }))
                    } else {
                        ExprKind::Call {
                            ty: self.typeck_results.node_type(fun.hir_id),
                            fun: self.mirror_expr(fun),
                            args: self.mirror_exprs(args),
                            from_hir_call: true,
                            fn_span: expr.span,
                        }
                    }
                }
            }

            hir::ExprKind::Use(expr, span) => {
                ExprKind::ByUse { expr: self.mirror_expr(expr), span }
            }

            hir::ExprKind::AddrOf(hir::BorrowKind::Ref, mutbl, arg) => {
                ExprKind::Borrow { borrow_kind: mutbl.to_borrow_kind(), arg: self.mirror_expr(arg) }
            }

            hir::ExprKind::AddrOf(hir::BorrowKind::Raw, mutability, arg) => {
                ExprKind::RawBorrow { mutability, arg: self.mirror_expr(arg) }
            }

            hir::ExprKind::Block(blk, _) => ExprKind::Block { block: self.mirror_block(blk) },

            hir::ExprKind::Assign(lhs, rhs, _) => {
                ExprKind::Assign { lhs: self.mirror_expr(lhs), rhs: self.mirror_expr(rhs) }
            }

            hir::ExprKind::AssignOp(op, lhs, rhs) => {
                if self.typeck_results.is_method_call(expr) {
                    let lhs = self.mirror_expr(lhs);
                    let rhs = self.mirror_expr(rhs);
                    self.overloaded_operator(expr, Box::new([lhs, rhs]))
                } else {
                    ExprKind::AssignOp {
                        op: assign_op(op.node),
                        lhs: self.mirror_expr(lhs),
                        rhs: self.mirror_expr(rhs),
                    }
                }
            }

            hir::ExprKind::Lit(lit) => ExprKind::Literal { lit, neg: false },

            hir::ExprKind::Binary(op, lhs, rhs) => {
                if self.typeck_results.is_method_call(expr) {
                    let lhs = self.mirror_expr(lhs);
                    let rhs = self.mirror_expr(rhs);
                    self.overloaded_operator(expr, Box::new([lhs, rhs]))
                } else {
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

            hir::ExprKind::Index(lhs, index, brackets_span) => {
                if self.typeck_results.is_method_call(expr) {
                    let lhs = self.mirror_expr(lhs);
                    let index = self.mirror_expr(index);
                    self.overloaded_place(
                        expr,
                        expr_ty,
                        None,
                        Box::new([lhs, index]),
                        brackets_span,
                    )
                } else {
                    ExprKind::Index { lhs: self.mirror_expr(lhs), index: self.mirror_expr(index) }
                }
            }

            hir::ExprKind::Unary(hir::UnOp::Deref, arg) => {
                if self.typeck_results.is_method_call(expr) {
                    let arg = self.mirror_expr(arg);
                    self.overloaded_place(expr, expr_ty, None, Box::new([arg]), expr.span)
                } else {
                    ExprKind::Deref { arg: self.mirror_expr(arg) }
                }
            }

            hir::ExprKind::Unary(hir::UnOp::Not, arg) => {
                if self.typeck_results.is_method_call(expr) {
                    let arg = self.mirror_expr(arg);
                    self.overloaded_operator(expr, Box::new([arg]))
                } else {
                    ExprKind::Unary { op: UnOp::Not, arg: self.mirror_expr(arg) }
                }
            }

            hir::ExprKind::Unary(hir::UnOp::Neg, arg) => {
                if self.typeck_results.is_method_call(expr) {
                    let arg = self.mirror_expr(arg);
                    self.overloaded_operator(expr, Box::new([arg]))
                } else if let hir::ExprKind::Lit(lit) = arg.kind {
                    ExprKind::Literal { lit, neg: true }
                } else {
                    ExprKind::Unary { op: UnOp::Neg, arg: self.mirror_expr(arg) }
                }
            }

            hir::ExprKind::Struct(qpath, fields, ref base) => match expr_ty.kind() {
                ty::Adt(adt, args) => match adt.adt_kind() {
                    AdtKind::Struct | AdtKind::Union => {
                        let user_provided_types = self.typeck_results.user_provided_types();
                        let user_ty = user_provided_types.get(expr.hir_id).copied().map(Box::new);
                        debug!("make_mirror_unadjusted: (struct/union) user_ty={:?}", user_ty);
                        ExprKind::Adt(Box::new(AdtExpr {
                            adt_def: *adt,
                            variant_index: FIRST_VARIANT,
                            args,
                            user_ty,
                            fields: self.field_refs(fields),
                            base: match base {
                                hir::StructTailExpr::Base(base) => AdtExprBase::Base(FruInfo {
                                    base: self.mirror_expr(base),
                                    field_types: self.typeck_results.fru_field_types()[expr.hir_id]
                                        .iter()
                                        .copied()
                                        .collect(),
                                }),
                                hir::StructTailExpr::DefaultFields(_) => {
                                    AdtExprBase::DefaultFields(
                                        self.typeck_results.fru_field_types()[expr.hir_id]
                                            .iter()
                                            .copied()
                                            .collect(),
                                    )
                                }
                                hir::StructTailExpr::None => AdtExprBase::None,
                            },
                        }))
                    }
                    AdtKind::Enum => {
                        let res = self.typeck_results.qpath_res(qpath, expr.hir_id);
                        match res {
                            Res::Def(DefKind::Variant, variant_id) => {
                                assert!(matches!(
                                    base,
                                    hir::StructTailExpr::None
                                        | hir::StructTailExpr::DefaultFields(_)
                                ));

                                let index = adt.variant_index_with_id(variant_id);
                                let user_provided_types = self.typeck_results.user_provided_types();
                                let user_ty =
                                    user_provided_types.get(expr.hir_id).copied().map(Box::new);
                                debug!("make_mirror_unadjusted: (variant) user_ty={:?}", user_ty);
                                ExprKind::Adt(Box::new(AdtExpr {
                                    adt_def: *adt,
                                    variant_index: index,
                                    args,
                                    user_ty,
                                    fields: self.field_refs(fields),
                                    base: match base {
                                        hir::StructTailExpr::DefaultFields(_) => {
                                            AdtExprBase::DefaultFields(
                                                self.typeck_results.fru_field_types()[expr.hir_id]
                                                    .iter()
                                                    .copied()
                                                    .collect(),
                                            )
                                        }
                                        hir::StructTailExpr::Base(base) => {
                                            span_bug!(base.span, "unexpected res: {:?}", res);
                                        }
                                        hir::StructTailExpr::None => AdtExprBase::None,
                                    },
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

            hir::ExprKind::Closure(hir::Closure { .. }) => {
                let closure_ty = self.typeck_results.expr_ty(expr);
                let (def_id, args, movability) = match *closure_ty.kind() {
                    ty::Closure(def_id, args) => (def_id, UpvarArgs::Closure(args), None),
                    ty::Coroutine(def_id, args) => {
                        (def_id, UpvarArgs::Coroutine(args), Some(tcx.coroutine_movability(def_id)))
                    }
                    ty::CoroutineClosure(def_id, args) => {
                        (def_id, UpvarArgs::CoroutineClosure(args), None)
                    }
                    _ => {
                        span_bug!(expr.span, "closure expr w/o closure type: {:?}", closure_ty);
                    }
                };
                let def_id = def_id.expect_local();

                let upvars = self
                    .tcx
                    .closure_captures(def_id)
                    .iter()
                    .zip_eq(args.upvar_tys())
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
                    args,
                    upvars,
                    movability,
                    fake_reads,
                }))
            }

            hir::ExprKind::Path(ref qpath) => {
                let res = self.typeck_results.qpath_res(qpath, expr.hir_id);
                self.convert_path_expr(expr, res)
            }

            hir::ExprKind::InlineAsm(asm) => ExprKind::InlineAsm(Box::new(InlineAsmExpr {
                asm_macro: asm.asm_macro,
                template: asm.template,
                operands: asm
                    .operands
                    .iter()
                    .map(|(op, _op_sp)| match *op {
                        hir::InlineAsmOperand::In { reg, expr } => {
                            InlineAsmOperand::In { reg, expr: self.mirror_expr(expr) }
                        }
                        hir::InlineAsmOperand::Out { reg, late, ref expr } => {
                            InlineAsmOperand::Out {
                                reg,
                                late,
                                expr: expr.map(|expr| self.mirror_expr(expr)),
                            }
                        }
                        hir::InlineAsmOperand::InOut { reg, late, expr } => {
                            InlineAsmOperand::InOut { reg, late, expr: self.mirror_expr(expr) }
                        }
                        hir::InlineAsmOperand::SplitInOut { reg, late, in_expr, ref out_expr } => {
                            InlineAsmOperand::SplitInOut {
                                reg,
                                late,
                                in_expr: self.mirror_expr(in_expr),
                                out_expr: out_expr.map(|expr| self.mirror_expr(expr)),
                            }
                        }
                        hir::InlineAsmOperand::Const { ref anon_const } => {
                            let ty = self.typeck_results.node_type(anon_const.hir_id);
                            let did = anon_const.def_id.to_def_id();
                            let typeck_root_def_id = tcx.typeck_root_def_id(did);
                            let parent_args = tcx.erase_regions(GenericArgs::identity_for_item(
                                tcx,
                                typeck_root_def_id,
                            ));
                            let args =
                                InlineConstArgs::new(tcx, InlineConstArgsParts { parent_args, ty })
                                    .args;

                            let uneval = mir::UnevaluatedConst::new(did, args);
                            let value = mir::Const::Unevaluated(uneval, ty);
                            InlineAsmOperand::Const { value, span: tcx.def_span(did) }
                        }
                        hir::InlineAsmOperand::SymFn { expr } => {
                            InlineAsmOperand::SymFn { value: self.mirror_expr(expr) }
                        }
                        hir::InlineAsmOperand::SymStatic { path: _, def_id } => {
                            InlineAsmOperand::SymStatic { def_id }
                        }
                        hir::InlineAsmOperand::Label { block } => {
                            InlineAsmOperand::Label { block: self.mirror_block(block) }
                        }
                    })
                    .collect(),
                options: asm.options,
                line_spans: asm.line_spans,
            })),

            hir::ExprKind::OffsetOf(_, _) => {
                let data = self.typeck_results.offset_of_data();
                let &(container, ref indices) = data.get(expr.hir_id).unwrap();
                let fields = tcx.mk_offset_of_from_iter(indices.iter().copied());

                ExprKind::OffsetOf { container, fields }
            }

            hir::ExprKind::ConstBlock(ref anon_const) => {
                let ty = self.typeck_results.node_type(anon_const.hir_id);
                let did = anon_const.def_id.to_def_id();
                let typeck_root_def_id = tcx.typeck_root_def_id(did);
                let parent_args =
                    tcx.erase_regions(GenericArgs::identity_for_item(tcx, typeck_root_def_id));
                let args = InlineConstArgs::new(tcx, InlineConstArgsParts { parent_args, ty }).args;

                ExprKind::ConstBlock { did, args }
            }
            // Now comes the rote stuff:
            hir::ExprKind::Repeat(v, _) => {
                let ty = self.typeck_results.expr_ty(expr);
                let ty::Array(_, count) = ty.kind() else {
                    span_bug!(expr.span, "unexpected repeat expr ty: {:?}", ty);
                };

                ExprKind::Repeat { value: self.mirror_expr(v), count: *count }
            }
            hir::ExprKind::Ret(v) => ExprKind::Return { value: v.map(|v| self.mirror_expr(v)) },
            hir::ExprKind::Become(call) => ExprKind::Become { value: self.mirror_expr(call) },
            hir::ExprKind::Break(dest, ref value) => match dest.target_id {
                Ok(target_id) => ExprKind::Break {
                    label: region::Scope {
                        local_id: target_id.local_id,
                        data: region::ScopeData::Node,
                    },
                    value: value.map(|value| self.mirror_expr(value)),
                },
                Err(err) => bug!("invalid loop id for break: {}", err),
            },
            hir::ExprKind::Continue(dest) => match dest.target_id {
                Ok(loop_id) => ExprKind::Continue {
                    label: region::Scope {
                        local_id: loop_id.local_id,
                        data: region::ScopeData::Node,
                    },
                },
                Err(err) => bug!("invalid loop id for continue: {}", err),
            },
            hir::ExprKind::Let(let_expr) => ExprKind::Let {
                expr: self.mirror_expr(let_expr.init),
                pat: self.pattern_from_hir(let_expr.pat),
            },
            hir::ExprKind::If(cond, then, else_opt) => ExprKind::If {
                if_then_scope: region::Scope {
                    local_id: then.hir_id.local_id,
                    data: {
                        if expr.span.at_least_rust_2024() {
                            region::ScopeData::IfThenRescope
                        } else {
                            region::ScopeData::IfThen
                        }
                    },
                },
                cond: self.mirror_expr(cond),
                then: self.mirror_expr(then),
                else_opt: else_opt.map(|el| self.mirror_expr(el)),
            },
            hir::ExprKind::Match(discr, arms, match_source) => ExprKind::Match {
                scrutinee: self.mirror_expr(discr),
                arms: arms.iter().map(|a| self.convert_arm(a)).collect(),
                match_source,
            },
            hir::ExprKind::Loop(body, ..) => {
                let block_ty = self.typeck_results.node_type(body.hir_id);
                let (temp_lifetime, backwards_incompatible) = self
                    .rvalue_scopes
                    .temporary_scope(self.region_scope_tree, body.hir_id.local_id);
                let block = self.mirror_block(body);
                let body = self.thir.exprs.push(Expr {
                    ty: block_ty,
                    temp_lifetime: TempLifetime { temp_lifetime, backwards_incompatible },
                    span: self.thir[block].span,
                    kind: ExprKind::Block { block },
                });
                ExprKind::Loop { body }
            }
            hir::ExprKind::Field(source, ..) => ExprKind::Field {
                lhs: self.mirror_expr(source),
                variant_index: FIRST_VARIANT,
                name: self.typeck_results.field_index(expr.hir_id),
            },
            hir::ExprKind::Cast(source, cast_ty) => {
                // Check for a user-given type annotation on this `cast`
                let user_provided_types = self.typeck_results.user_provided_types();
                let user_ty = user_provided_types.get(cast_ty.hir_id);

                debug!(
                    "cast({:?}) has ty w/ hir_id {:?} and user provided ty {:?}",
                    expr, cast_ty.hir_id, user_ty,
                );

                let cast = self.mirror_expr_cast(
                    source,
                    TempLifetime { temp_lifetime, backwards_incompatible },
                    expr.span,
                );

                if let Some(user_ty) = user_ty {
                    // NOTE: Creating a new Expr and wrapping a Cast inside of it may be
                    //       inefficient, revisit this when performance becomes an issue.
                    let cast_expr = self.thir.exprs.push(Expr {
                        temp_lifetime: TempLifetime { temp_lifetime, backwards_incompatible },
                        ty: expr_ty,
                        span: expr.span,
                        kind: cast,
                    });
                    debug!("make_mirror_unadjusted: (cast) user_ty={:?}", user_ty);

                    ExprKind::ValueTypeAscription {
                        source: cast_expr,
                        user_ty: Some(Box::new(*user_ty)),
                        user_ty_span: cast_ty.span,
                    }
                } else {
                    cast
                }
            }
            hir::ExprKind::Type(source, ty) => {
                let user_provided_types = self.typeck_results.user_provided_types();
                let user_ty = user_provided_types.get(ty.hir_id).copied().map(Box::new);
                debug!("make_mirror_unadjusted: (type) user_ty={:?}", user_ty);
                let mirrored = self.mirror_expr(source);
                if source.is_syntactic_place_expr() {
                    ExprKind::PlaceTypeAscription {
                        source: mirrored,
                        user_ty,
                        user_ty_span: ty.span,
                    }
                } else {
                    ExprKind::ValueTypeAscription {
                        source: mirrored,
                        user_ty,
                        user_ty_span: ty.span,
                    }
                }
            }

            hir::ExprKind::UnsafeBinderCast(UnsafeBinderCastKind::Unwrap, source, _ty) => {
                // FIXME(unsafe_binders): Take into account the ascribed type, too.
                let mirrored = self.mirror_expr(source);
                if source.is_syntactic_place_expr() {
                    ExprKind::PlaceUnwrapUnsafeBinder { source: mirrored }
                } else {
                    ExprKind::ValueUnwrapUnsafeBinder { source: mirrored }
                }
            }
            hir::ExprKind::UnsafeBinderCast(UnsafeBinderCastKind::Wrap, source, _ty) => {
                // FIXME(unsafe_binders): Take into account the ascribed type, too.
                let mirrored = self.mirror_expr(source);
                ExprKind::WrapUnsafeBinder { source: mirrored }
            }

            hir::ExprKind::DropTemps(source) => ExprKind::Use { source: self.mirror_expr(source) },
            hir::ExprKind::Array(fields) => ExprKind::Array { fields: self.mirror_exprs(fields) },
            hir::ExprKind::Tup(fields) => ExprKind::Tuple { fields: self.mirror_exprs(fields) },

            hir::ExprKind::Yield(v, _) => ExprKind::Yield { value: self.mirror_expr(v) },
            hir::ExprKind::Err(_) => unreachable!("cannot lower a `hir::ExprKind::Err` to THIR"),
        };

        Expr {
            temp_lifetime: TempLifetime { temp_lifetime, backwards_incompatible },
            ty: expr_ty,
            span: expr.span,
            kind,
        }
    }

    fn user_args_applied_to_res(
        &mut self,
        hir_id: hir::HirId,
        res: Res,
    ) -> Option<Box<ty::CanonicalUserType<'tcx>>> {
        debug!("user_args_applied_to_res: res={:?}", res);
        let user_provided_type = match res {
            // A reference to something callable -- e.g., a fn, method, or
            // a tuple-struct or tuple-variant. This has the type of a
            // `Fn` but with the user-given generic parameters.
            Res::Def(DefKind::Fn, _)
            | Res::Def(DefKind::AssocFn, _)
            | Res::Def(DefKind::Ctor(_, CtorKind::Fn), _)
            | Res::Def(DefKind::Const, _)
            | Res::Def(DefKind::AssocConst, _) => {
                self.typeck_results.user_provided_types().get(hir_id).copied().map(Box::new)
            }

            // A unit struct/variant which is used as a value (e.g.,
            // `None`). This has the type of the enum/struct that defines
            // this variant -- but with the generic parameters given by the
            // user.
            Res::Def(DefKind::Ctor(_, CtorKind::Const), _) => {
                self.user_args_applied_to_ty_of_hir_id(hir_id).map(Box::new)
            }

            // `Self` is used in expression as a tuple struct constructor or a unit struct constructor
            Res::SelfCtor(_) => self.user_args_applied_to_ty_of_hir_id(hir_id).map(Box::new),

            _ => bug!("user_args_applied_to_res: unexpected res {:?} at {:?}", res, hir_id),
        };
        debug!("user_args_applied_to_res: user_provided_type={:?}", user_provided_type);
        user_provided_type
    }

    fn method_callee(
        &mut self,
        expr: &hir::Expr<'_>,
        span: Span,
        overloaded_callee: Option<Ty<'tcx>>,
    ) -> Expr<'tcx> {
        let (temp_lifetime, backwards_incompatible) =
            self.rvalue_scopes.temporary_scope(self.region_scope_tree, expr.hir_id.local_id);
        let (ty, user_ty) = match overloaded_callee {
            Some(fn_def) => (fn_def, None),
            None => {
                let (kind, def_id) =
                    self.typeck_results.type_dependent_def(expr.hir_id).unwrap_or_else(|| {
                        span_bug!(expr.span, "no type-dependent def for method callee")
                    });
                let user_ty = self.user_args_applied_to_res(expr.hir_id, Res::Def(kind, def_id));
                debug!("method_callee: user_ty={:?}", user_ty);
                (
                    Ty::new_fn_def(self.tcx, def_id, self.typeck_results.node_args(expr.hir_id)),
                    user_ty,
                )
            }
        };
        Expr {
            temp_lifetime: TempLifetime { temp_lifetime, backwards_incompatible },
            ty,
            span,
            kind: ExprKind::ZstLiteral { user_ty },
        }
    }

    fn convert_arm(&mut self, arm: &'tcx hir::Arm<'tcx>) -> ArmId {
        let arm = Arm {
            pattern: self.pattern_from_hir(&arm.pat),
            guard: arm.guard.as_ref().map(|g| self.mirror_expr(g)),
            body: self.mirror_expr(arm.body),
            lint_level: LintLevel::Explicit(arm.hir_id),
            scope: region::Scope { local_id: arm.hir_id.local_id, data: region::ScopeData::Node },
            span: arm.span,
        };
        self.thir.arms.push(arm)
    }

    fn convert_path_expr(&mut self, expr: &'tcx hir::Expr<'tcx>, res: Res) -> ExprKind<'tcx> {
        let args = self.typeck_results.node_args(expr.hir_id);
        match res {
            // A regular function, constructor function or a constant.
            Res::Def(DefKind::Fn, _)
            | Res::Def(DefKind::AssocFn, _)
            | Res::Def(DefKind::Ctor(_, CtorKind::Fn), _)
            | Res::SelfCtor(_) => {
                let user_ty = self.user_args_applied_to_res(expr.hir_id, res);
                ExprKind::ZstLiteral { user_ty }
            }

            Res::Def(DefKind::ConstParam, def_id) => {
                let hir_id = self.tcx.local_def_id_to_hir_id(def_id.expect_local());
                let generics = self.tcx.generics_of(hir_id.owner);
                let Some(&index) = generics.param_def_id_to_index.get(&def_id) else {
                    span_bug!(
                        expr.span,
                        "Should have already errored about late bound consts: {def_id:?}"
                    );
                };
                let name = self.tcx.hir_name(hir_id);
                let param = ty::ParamConst::new(index, name);

                ExprKind::ConstParam { param, def_id }
            }

            Res::Def(DefKind::Const, def_id) | Res::Def(DefKind::AssocConst, def_id) => {
                let user_ty = self.user_args_applied_to_res(expr.hir_id, res);
                ExprKind::NamedConst { def_id, args, user_ty }
            }

            Res::Def(DefKind::Ctor(_, CtorKind::Const), def_id) => {
                let user_provided_types = self.typeck_results.user_provided_types();
                let user_ty = user_provided_types.get(expr.hir_id).copied().map(Box::new);
                debug!("convert_path_expr: user_ty={:?}", user_ty);
                let ty = self.typeck_results.node_type(expr.hir_id);
                match ty.kind() {
                    // A unit struct/variant which is used as a value.
                    // We return a completely different ExprKind here to account for this special case.
                    ty::Adt(adt_def, args) => ExprKind::Adt(Box::new(AdtExpr {
                        adt_def: *adt_def,
                        variant_index: adt_def.variant_index_with_ctor_id(def_id),
                        args,
                        user_ty,
                        fields: Box::new([]),
                        base: AdtExprBase::None,
                    })),
                    _ => bug!("unexpected ty: {:?}", ty),
                }
            }

            // A source Rust `path::to::STATIC` is a place expr like *&ident is.
            // In THIR, we make them exactly equivalent by inserting the implied *& or *&raw,
            // but distinguish between &STATIC and &THREAD_LOCAL as they have different semantics
            Res::Def(DefKind::Static { .. }, id) => {
                // this is &raw for extern static or static mut, and & for other statics
                let ty = self.tcx.static_ptr_ty(id, self.typing_env);
                let (temp_lifetime, backwards_incompatible) = self
                    .rvalue_scopes
                    .temporary_scope(self.region_scope_tree, expr.hir_id.local_id);
                let kind = if self.tcx.is_thread_local_static(id) {
                    ExprKind::ThreadLocalRef(id)
                } else {
                    let alloc_id = self.tcx.reserve_and_set_static_alloc(id);
                    ExprKind::StaticRef { alloc_id, ty, def_id: id }
                };
                ExprKind::Deref {
                    arg: self.thir.exprs.push(Expr {
                        ty,
                        temp_lifetime: TempLifetime { temp_lifetime, backwards_incompatible },
                        span: expr.span,
                        kind,
                    }),
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
            .is_some_and(|upvars| upvars.contains_key(&var_hir_id));

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
        overloaded_callee: Option<Ty<'tcx>>,
        args: Box<[ExprId]>,
        span: Span,
    ) -> ExprKind<'tcx> {
        // For an overloaded *x or x[y] expression of type T, the method
        // call returns an &T and we must add the deref so that the types
        // line up (this is because `*x` and `x[y]` represent places):

        // Reconstruct the output assuming it's a reference with the
        // same region and mutability as the receiver. This holds for
        // `Deref(Mut)::deref(_mut)` and `Index(Mut)::index(_mut)`.
        let ty::Ref(region, _, mutbl) = *self.thir[args[0]].ty.kind() else {
            span_bug!(span, "overloaded_place: receiver is not a reference");
        };
        let ref_ty = Ty::new_ref(self.tcx, region, place_ty, mutbl);

        // construct the complete expression `foo()` for the overloaded call,
        // which will yield the &T type
        let (temp_lifetime, backwards_incompatible) =
            self.rvalue_scopes.temporary_scope(self.region_scope_tree, expr.hir_id.local_id);
        let fun = self.method_callee(expr, span, overloaded_callee);
        let fun = self.thir.exprs.push(fun);
        let fun_ty = self.thir[fun].ty;
        let ref_expr = self.thir.exprs.push(Expr {
            temp_lifetime: TempLifetime { temp_lifetime, backwards_incompatible },
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
        let (temp_lifetime, backwards_incompatible) = self
            .rvalue_scopes
            .temporary_scope(self.region_scope_tree, closure_expr.hir_id.local_id);
        let var_ty = place.base_ty;

        // The result of capture analysis in `rustc_hir_typeck/src/upvar.rs` represents a captured path
        // as it's seen for use within the closure and not at the time of closure creation.
        //
        // That is we see expect to see it start from a captured upvar and not something that is local
        // to the closure's parent.
        let var_hir_id = match place.base {
            HirPlaceBase::Upvar(upvar_id) => upvar_id.var_path.hir_id,
            base => bug!("Expected an upvar, found {:?}", base),
        };

        let mut captured_place_expr = Expr {
            temp_lifetime: TempLifetime { temp_lifetime, backwards_incompatible },
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
                    name: field,
                },
                HirProjectionKind::OpaqueCast => {
                    ExprKind::Use { source: self.thir.exprs.push(captured_place_expr) }
                }
                HirProjectionKind::UnwrapUnsafeBinder => ExprKind::PlaceUnwrapUnsafeBinder {
                    source: self.thir.exprs.push(captured_place_expr),
                },
                HirProjectionKind::Index | HirProjectionKind::Subslice => {
                    // We don't capture these projections, so we can ignore them here
                    continue;
                }
            };

            captured_place_expr = Expr {
                temp_lifetime: TempLifetime { temp_lifetime, backwards_incompatible },
                ty: proj.ty,
                span: closure_expr.span,
                kind,
            };
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
        let (temp_lifetime, backwards_incompatible) = self
            .rvalue_scopes
            .temporary_scope(self.region_scope_tree, closure_expr.hir_id.local_id);

        match upvar_capture {
            ty::UpvarCapture::ByValue => captured_place_expr,
            ty::UpvarCapture::ByUse => {
                let span = captured_place_expr.span;
                let expr_id = self.thir.exprs.push(captured_place_expr);

                Expr {
                    temp_lifetime: TempLifetime { temp_lifetime, backwards_incompatible },
                    ty: upvar_ty,
                    span: closure_expr.span,
                    kind: ExprKind::ByUse { expr: expr_id, span },
                }
            }
            ty::UpvarCapture::ByRef(upvar_borrow) => {
                let borrow_kind = match upvar_borrow {
                    ty::BorrowKind::Immutable => BorrowKind::Shared,
                    ty::BorrowKind::UniqueImmutable => {
                        BorrowKind::Mut { kind: mir::MutBorrowKind::ClosureCapture }
                    }
                    ty::BorrowKind::Mutable => {
                        BorrowKind::Mut { kind: mir::MutBorrowKind::Default }
                    }
                };
                Expr {
                    temp_lifetime: TempLifetime { temp_lifetime, backwards_incompatible },
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
                name: self.typeck_results.field_index(field.hir_id),
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
                kind: match allow_two_phase_borrow {
                    AllowTwoPhase::Yes => mir::MutBorrowKind::TwoPhaseBorrow,
                    AllowTwoPhase::No => mir::MutBorrowKind::Default,
                },
            },
            AutoBorrowMutability::Not => BorrowKind::Shared,
        }
    }
}

impl ToBorrowKind for hir::Mutability {
    fn to_borrow_kind(&self) -> BorrowKind {
        match *self {
            hir::Mutability::Mut => BorrowKind::Mut { kind: mir::MutBorrowKind::Default },
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

fn assign_op(op: hir::AssignOpKind) -> AssignOp {
    match op {
        hir::AssignOpKind::AddAssign => AssignOp::AddAssign,
        hir::AssignOpKind::SubAssign => AssignOp::SubAssign,
        hir::AssignOpKind::MulAssign => AssignOp::MulAssign,
        hir::AssignOpKind::DivAssign => AssignOp::DivAssign,
        hir::AssignOpKind::RemAssign => AssignOp::RemAssign,
        hir::AssignOpKind::BitXorAssign => AssignOp::BitXorAssign,
        hir::AssignOpKind::BitAndAssign => AssignOp::BitAndAssign,
        hir::AssignOpKind::BitOrAssign => AssignOp::BitOrAssign,
        hir::AssignOpKind::ShlAssign => AssignOp::ShlAssign,
        hir::AssignOpKind::ShrAssign => AssignOp::ShrAssign,
    }
}
