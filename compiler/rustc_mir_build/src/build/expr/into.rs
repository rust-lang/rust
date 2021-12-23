//! See docs in build/expr/mod.rs

use crate::build::expr::category::{Category, RvalueFunc};
use crate::build::{BlockAnd, BlockAndExtension, BlockFrame, Builder};
use rustc_ast::InlineAsmOptions;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_hir as hir;
use rustc_index::vec::Idx;
use rustc_middle::mir::*;
use rustc_middle::thir::*;
use rustc_middle::ty::{self, CanonicalUserTypeAnnotation};
use std::iter;

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Compile `expr`, storing the result into `destination`, which
    /// is assumed to be uninitialized.
    crate fn expr_into_dest(
        &mut self,
        destination: Place<'tcx>,
        mut block: BasicBlock,
        expr: &Expr<'tcx>,
    ) -> BlockAnd<()> {
        debug!("expr_into_dest(destination={:?}, block={:?}, expr={:?})", destination, block, expr);

        // since we frequently have to reference `self` from within a
        // closure, where `self` would be shadowed, it's easier to
        // just use the name `this` uniformly
        let this = self;
        let expr_span = expr.span;
        let source_info = this.source_info(expr_span);

        let expr_is_block_or_scope =
            matches!(expr.kind, ExprKind::Block { .. } | ExprKind::Scope { .. });

        if !expr_is_block_or_scope {
            this.block_context.push(BlockFrame::SubExpr);
        }

        let block_and = match expr.kind {
            ExprKind::Scope { region_scope, lint_level, value } => {
                let region_scope = (region_scope, source_info);
                ensure_sufficient_stack(|| {
                    this.in_scope(region_scope, lint_level, |this| {
                        this.expr_into_dest(destination, block, &this.thir[value])
                    })
                })
            }
            ExprKind::Block { body: ref ast_block } => {
                this.ast_block(destination, block, ast_block, source_info)
            }
            ExprKind::Match { scrutinee, ref arms } => {
                this.match_expr(destination, expr_span, block, &this.thir[scrutinee], arms)
            }
            ExprKind::If { cond, then, else_opt, if_then_scope } => {
                let then_blk;
                let then_expr = &this.thir[then];
                let then_source_info = this.source_info(then_expr.span);
                let condition_scope = this.local_scope();

                let mut else_blk = unpack!(
                    then_blk = this.in_scope(
                        (if_then_scope, then_source_info),
                        LintLevel::Inherited,
                        |this| {
                            let (then_block, else_block) =
                                this.in_if_then_scope(condition_scope, |this| {
                                    let then_blk = unpack!(this.then_else_break(
                                        block,
                                        &this.thir[cond],
                                        Some(condition_scope),
                                        condition_scope,
                                        then_expr.span,
                                    ));
                                    this.expr_into_dest(destination, then_blk, then_expr)
                                });
                            then_block.and(else_block)
                        },
                    )
                );

                else_blk = if let Some(else_opt) = else_opt {
                    unpack!(this.expr_into_dest(destination, else_blk, &this.thir[else_opt]))
                } else {
                    // Body of the `if` expression without an `else` clause must return `()`, thus
                    // we implicitly generate an `else {}` if it is not specified.
                    let correct_si = this.source_info(expr_span.shrink_to_hi());
                    this.cfg.push_assign_unit(else_blk, correct_si, destination, this.tcx);
                    else_blk
                };

                let join_block = this.cfg.start_new_block();
                this.cfg.terminate(
                    then_blk,
                    source_info,
                    TerminatorKind::Goto { target: join_block },
                );
                this.cfg.terminate(
                    else_blk,
                    source_info,
                    TerminatorKind::Goto { target: join_block },
                );

                join_block.unit()
            }
            ExprKind::Let { expr, ref pat } => {
                let scope = this.local_scope();
                let (true_block, false_block) = this.in_if_then_scope(scope, |this| {
                    this.lower_let_expr(block, &this.thir[expr], pat, scope, expr_span)
                });

                let join_block = this.cfg.start_new_block();

                this.cfg.push_assign_constant(
                    true_block,
                    source_info,
                    destination,
                    Constant {
                        span: expr_span,
                        user_ty: None,
                        literal: ty::Const::from_bool(this.tcx, true).into(),
                    },
                );

                this.cfg.push_assign_constant(
                    false_block,
                    source_info,
                    destination,
                    Constant {
                        span: expr_span,
                        user_ty: None,
                        literal: ty::Const::from_bool(this.tcx, false).into(),
                    },
                );

                this.cfg.goto(true_block, source_info, join_block);
                this.cfg.goto(false_block, source_info, join_block);
                join_block.unit()
            }
            ExprKind::NeverToAny { source } => {
                let source = &this.thir[source];
                let is_call =
                    matches!(source.kind, ExprKind::Call { .. } | ExprKind::InlineAsm { .. });

                // (#66975) Source could be a const of type `!`, so has to
                // exist in the generated MIR.
                unpack!(
                    block = this.as_temp(block, Some(this.local_scope()), source, Mutability::Mut,)
                );

                // This is an optimization. If the expression was a call then we already have an
                // unreachable block. Don't bother to terminate it and create a new one.
                if is_call {
                    block.unit()
                } else {
                    this.cfg.terminate(block, source_info, TerminatorKind::Unreachable);
                    let end_block = this.cfg.start_new_block();
                    end_block.unit()
                }
            }
            ExprKind::LogicalOp { op, lhs, rhs } => {
                // And:
                //
                // [block: If(lhs)] -true-> [else_block: dest = (rhs)]
                //        | (false)
                //  [shortcurcuit_block: dest = false]
                //
                // Or:
                //
                // [block: If(lhs)] -false-> [else_block: dest = (rhs)]
                //        | (true)
                //  [shortcurcuit_block: dest = true]

                let (shortcircuit_block, mut else_block, join_block) = (
                    this.cfg.start_new_block(),
                    this.cfg.start_new_block(),
                    this.cfg.start_new_block(),
                );

                let lhs = unpack!(block = this.as_local_operand(block, &this.thir[lhs]));
                let blocks = match op {
                    LogicalOp::And => (else_block, shortcircuit_block),
                    LogicalOp::Or => (shortcircuit_block, else_block),
                };
                let term = TerminatorKind::if_(this.tcx, lhs, blocks.0, blocks.1);
                this.cfg.terminate(block, source_info, term);

                this.cfg.push_assign_constant(
                    shortcircuit_block,
                    source_info,
                    destination,
                    Constant {
                        span: expr_span,
                        user_ty: None,
                        literal: match op {
                            LogicalOp::And => ty::Const::from_bool(this.tcx, false).into(),
                            LogicalOp::Or => ty::Const::from_bool(this.tcx, true).into(),
                        },
                    },
                );
                this.cfg.goto(shortcircuit_block, source_info, join_block);

                let rhs = unpack!(else_block = this.as_local_operand(else_block, &this.thir[rhs]));
                this.cfg.push_assign(else_block, source_info, destination, Rvalue::Use(rhs));
                this.cfg.goto(else_block, source_info, join_block);

                join_block.unit()
            }
            ExprKind::Loop { body } => {
                // [block]
                //    |
                //   [loop_block] -> [body_block] -/eval. body/-> [body_block_end]
                //    |        ^                                         |
                // false link  |                                         |
                //    |        +-----------------------------------------+
                //    +-> [diverge_cleanup]
                // The false link is required to make sure borrowck considers unwinds through the
                // body, even when the exact code in the body cannot unwind

                let loop_block = this.cfg.start_new_block();

                // Start the loop.
                this.cfg.goto(block, source_info, loop_block);

                this.in_breakable_scope(Some(loop_block), destination, expr_span, move |this| {
                    // conduct the test, if necessary
                    let body_block = this.cfg.start_new_block();
                    this.cfg.terminate(
                        loop_block,
                        source_info,
                        TerminatorKind::FalseUnwind { real_target: body_block, unwind: None },
                    );
                    this.diverge_from(loop_block);

                    // The “return” value of the loop body must always be a unit. We therefore
                    // introduce a unit temporary as the destination for the loop body.
                    let tmp = this.get_unit_temp();
                    // Execute the body, branching back to the test.
                    let body_block_end =
                        unpack!(this.expr_into_dest(tmp, body_block, &this.thir[body]));
                    this.cfg.goto(body_block_end, source_info, loop_block);

                    // Loops are only exited by `break` expressions.
                    None
                })
            }
            ExprKind::Call { ty: _, fun, ref args, from_hir_call, fn_span } => {
                let fun = unpack!(block = this.as_local_operand(block, &this.thir[fun]));
                let args: Vec<_> = args
                    .into_iter()
                    .copied()
                    .map(|arg| unpack!(block = this.as_local_call_operand(block, &this.thir[arg])))
                    .collect();

                let success = this.cfg.start_new_block();

                this.record_operands_moved(&args);

                debug!("expr_into_dest: fn_span={:?}", fn_span);

                this.cfg.terminate(
                    block,
                    source_info,
                    TerminatorKind::Call {
                        func: fun,
                        args,
                        cleanup: None,
                        // FIXME(varkor): replace this with an uninhabitedness-based check.
                        // This requires getting access to the current module to call
                        // `tcx.is_ty_uninhabited_from`, which is currently tricky to do.
                        destination: if expr.ty.is_never() {
                            None
                        } else {
                            Some((destination, success))
                        },
                        from_hir_call,
                        fn_span,
                    },
                );
                this.diverge_from(block);
                success.unit()
            }
            ExprKind::Use { source } => this.expr_into_dest(destination, block, &this.thir[source]),
            ExprKind::Borrow { arg, borrow_kind } => {
                let arg = &this.thir[arg];
                // We don't do this in `as_rvalue` because we use `as_place`
                // for borrow expressions, so we cannot create an `RValue` that
                // remains valid across user code. `as_rvalue` is usually called
                // by this method anyway, so this shouldn't cause too many
                // unnecessary temporaries.
                let arg_place = match borrow_kind {
                    BorrowKind::Shared => unpack!(block = this.as_read_only_place(block, arg)),
                    _ => unpack!(block = this.as_place(block, arg)),
                };
                let borrow = Rvalue::Ref(this.tcx.lifetimes.re_erased, borrow_kind, arg_place);
                this.cfg.push_assign(block, source_info, destination, borrow);
                block.unit()
            }
            ExprKind::AddressOf { mutability, arg } => {
                let arg = &this.thir[arg];
                let place = match mutability {
                    hir::Mutability::Not => this.as_read_only_place(block, arg),
                    hir::Mutability::Mut => this.as_place(block, arg),
                };
                let address_of = Rvalue::AddressOf(mutability, unpack!(block = place));
                this.cfg.push_assign(block, source_info, destination, address_of);
                block.unit()
            }
            ExprKind::Adt(box Adt {
                adt_def,
                variant_index,
                substs,
                user_ty,
                ref fields,
                ref base,
            }) => {
                // See the notes for `ExprKind::Array` in `as_rvalue` and for
                // `ExprKind::Borrow` above.
                let is_union = adt_def.is_union();
                let active_field_index = if is_union { Some(fields[0].name.index()) } else { None };

                let scope = this.local_scope();

                // first process the set of fields that were provided
                // (evaluating them in order given by user)
                let fields_map: FxHashMap<_, _> = fields
                    .into_iter()
                    .map(|f| {
                        let local_info = Box::new(LocalInfo::AggregateTemp);
                        (
                            f.name,
                            unpack!(
                                block = this.as_operand(
                                    block,
                                    Some(scope),
                                    &this.thir[f.expr],
                                    Some(local_info)
                                )
                            ),
                        )
                    })
                    .collect();

                let field_names: Vec<_> =
                    (0..adt_def.variants[variant_index].fields.len()).map(Field::new).collect();

                let fields: Vec<_> = if let Some(FruInfo { base, field_types }) = base {
                    let place_builder =
                        unpack!(block = this.as_place_builder(block, &this.thir[*base]));

                    // MIR does not natively support FRU, so for each
                    // base-supplied field, generate an operand that
                    // reads it from the base.
                    iter::zip(field_names, &**field_types)
                        .map(|(n, ty)| match fields_map.get(&n) {
                            Some(v) => v.clone(),
                            None => {
                                let place_builder = place_builder.clone();
                                this.consume_by_copy_or_move(
                                    place_builder
                                        .field(n, ty)
                                        .into_place(this.tcx, this.typeck_results),
                                )
                            }
                        })
                        .collect()
                } else {
                    field_names.iter().filter_map(|n| fields_map.get(n).cloned()).collect()
                };

                let inferred_ty = expr.ty;
                let user_ty = user_ty.map(|ty| {
                    this.canonical_user_type_annotations.push(CanonicalUserTypeAnnotation {
                        span: source_info.span,
                        user_ty: ty,
                        inferred_ty,
                    })
                });
                let adt = Box::new(AggregateKind::Adt(
                    adt_def.did,
                    variant_index,
                    substs,
                    user_ty,
                    active_field_index,
                ));
                this.cfg.push_assign(
                    block,
                    source_info,
                    destination,
                    Rvalue::Aggregate(adt, fields),
                );
                block.unit()
            }
            ExprKind::InlineAsm { template, ref operands, options, line_spans } => {
                use rustc_middle::{mir, thir};
                let operands = operands
                    .into_iter()
                    .map(|op| match *op {
                        thir::InlineAsmOperand::In { reg, expr } => mir::InlineAsmOperand::In {
                            reg,
                            value: unpack!(block = this.as_local_operand(block, &this.thir[expr])),
                        },
                        thir::InlineAsmOperand::Out { reg, late, expr } => {
                            mir::InlineAsmOperand::Out {
                                reg,
                                late,
                                place: expr.map(|expr| {
                                    unpack!(block = this.as_place(block, &this.thir[expr]))
                                }),
                            }
                        }
                        thir::InlineAsmOperand::InOut { reg, late, expr } => {
                            let place = unpack!(block = this.as_place(block, &this.thir[expr]));
                            mir::InlineAsmOperand::InOut {
                                reg,
                                late,
                                // This works because asm operands must be Copy
                                in_value: Operand::Copy(place),
                                out_place: Some(place),
                            }
                        }
                        thir::InlineAsmOperand::SplitInOut { reg, late, in_expr, out_expr } => {
                            mir::InlineAsmOperand::InOut {
                                reg,
                                late,
                                in_value: unpack!(
                                    block = this.as_local_operand(block, &this.thir[in_expr])
                                ),
                                out_place: out_expr.map(|out_expr| {
                                    unpack!(block = this.as_place(block, &this.thir[out_expr]))
                                }),
                            }
                        }
                        thir::InlineAsmOperand::Const { value, span } => {
                            mir::InlineAsmOperand::Const {
                                value: Box::new(Constant {
                                    span,
                                    user_ty: None,
                                    literal: value.into(),
                                }),
                            }
                        }
                        thir::InlineAsmOperand::SymFn { expr } => mir::InlineAsmOperand::SymFn {
                            value: Box::new(this.as_constant(&this.thir[expr])),
                        },
                        thir::InlineAsmOperand::SymStatic { def_id } => {
                            mir::InlineAsmOperand::SymStatic { def_id }
                        }
                    })
                    .collect();

                if !options.contains(InlineAsmOptions::NORETURN) {
                    this.cfg.push_assign_unit(block, source_info, destination, this.tcx);
                }

                let destination_block = this.cfg.start_new_block();
                this.cfg.terminate(
                    block,
                    source_info,
                    TerminatorKind::InlineAsm {
                        template,
                        operands,
                        options,
                        line_spans,
                        destination: if options.contains(InlineAsmOptions::NORETURN) {
                            None
                        } else {
                            Some(destination_block)
                        },
                        cleanup: None,
                    },
                );
                if options.contains(InlineAsmOptions::MAY_UNWIND) {
                    this.diverge_from(block);
                }
                destination_block.unit()
            }

            // These cases don't actually need a destination
            ExprKind::Assign { .. }
            | ExprKind::AssignOp { .. }
            | ExprKind::LlvmInlineAsm { .. } => {
                unpack!(block = this.stmt_expr(block, expr, None));
                this.cfg.push_assign_unit(block, source_info, destination, this.tcx);
                block.unit()
            }

            ExprKind::Continue { .. } | ExprKind::Break { .. } | ExprKind::Return { .. } => {
                unpack!(block = this.stmt_expr(block, expr, None));
                // No assign, as these have type `!`.
                block.unit()
            }

            // Avoid creating a temporary
            ExprKind::VarRef { .. }
            | ExprKind::UpvarRef { .. }
            | ExprKind::PlaceTypeAscription { .. }
            | ExprKind::ValueTypeAscription { .. } => {
                debug_assert!(Category::of(&expr.kind) == Some(Category::Place));

                let place = unpack!(block = this.as_place(block, expr));
                let rvalue = Rvalue::Use(this.consume_by_copy_or_move(place));
                this.cfg.push_assign(block, source_info, destination, rvalue);
                block.unit()
            }
            ExprKind::Index { .. } | ExprKind::Deref { .. } | ExprKind::Field { .. } => {
                debug_assert_eq!(Category::of(&expr.kind), Some(Category::Place));

                // Create a "fake" temporary variable so that we check that the
                // value is Sized. Usually, this is caught in type checking, but
                // in the case of box expr there is no such check.
                if !destination.projection.is_empty() {
                    this.local_decls.push(LocalDecl::new(expr.ty, expr.span));
                }

                let place = unpack!(block = this.as_place(block, expr));
                let rvalue = Rvalue::Use(this.consume_by_copy_or_move(place));
                this.cfg.push_assign(block, source_info, destination, rvalue);
                block.unit()
            }

            ExprKind::Yield { value } => {
                let scope = this.local_scope();
                let value =
                    unpack!(block = this.as_operand(block, Some(scope), &this.thir[value], None));
                let resume = this.cfg.start_new_block();
                this.cfg.terminate(
                    block,
                    source_info,
                    TerminatorKind::Yield { value, resume, resume_arg: destination, drop: None },
                );
                this.generator_drop_cleanup(block);
                resume.unit()
            }

            // these are the cases that are more naturally handled by some other mode
            ExprKind::Unary { .. }
            | ExprKind::Binary { .. }
            | ExprKind::Box { .. }
            | ExprKind::Cast { .. }
            | ExprKind::Pointer { .. }
            | ExprKind::Repeat { .. }
            | ExprKind::Array { .. }
            | ExprKind::Tuple { .. }
            | ExprKind::Closure { .. }
            | ExprKind::ConstBlock { .. }
            | ExprKind::Literal { .. }
            | ExprKind::ThreadLocalRef(_)
            | ExprKind::StaticRef { .. } => {
                debug_assert!(match Category::of(&expr.kind).unwrap() {
                    // should be handled above
                    Category::Rvalue(RvalueFunc::Into) => false,

                    // must be handled above or else we get an
                    // infinite loop in the builder; see
                    // e.g., `ExprKind::VarRef` above
                    Category::Place => false,

                    _ => true,
                });

                let rvalue = unpack!(block = this.as_local_rvalue(block, expr));
                this.cfg.push_assign(block, source_info, destination, rvalue);
                block.unit()
            }
        };

        if !expr_is_block_or_scope {
            let popped = this.block_context.pop();
            assert!(popped.is_some());
        }

        block_and
    }
}
