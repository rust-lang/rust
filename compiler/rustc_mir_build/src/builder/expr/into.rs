//! See docs in build/expr/mod.rs

use rustc_ast::{AsmMacro, InlineAsmOptions};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_hir as hir;
use rustc_hir::lang_items::LangItem;
use rustc_middle::mir::*;
use rustc_middle::span_bug;
use rustc_middle::thir::*;
use rustc_middle::ty::{CanonicalUserTypeAnnotation, Ty};
use rustc_span::DUMMY_SP;
use rustc_span::source_map::Spanned;
use rustc_trait_selection::infer::InferCtxtExt;
use tracing::{debug, instrument};

use crate::builder::expr::category::{Category, RvalueFunc};
use crate::builder::matches::DeclareLetBindings;
use crate::builder::{BlockAnd, BlockAndExtension, BlockFrame, Builder, NeedsTemporary};

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Compile `expr`, storing the result into `destination`, which
    /// is assumed to be uninitialized.
    #[instrument(level = "debug", skip(self))]
    pub(crate) fn expr_into_dest(
        &mut self,
        destination: Place<'tcx>,
        mut block: BasicBlock,
        expr_id: ExprId,
    ) -> BlockAnd<()> {
        // since we frequently have to reference `self` from within a
        // closure, where `self` would be shadowed, it's easier to
        // just use the name `this` uniformly
        let this = self;
        let expr = &this.thir[expr_id];
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
                        this.expr_into_dest(destination, block, value)
                    })
                })
            }
            ExprKind::Block { block: ast_block } => {
                this.ast_block(destination, block, ast_block, source_info)
            }
            ExprKind::Match { scrutinee, ref arms, .. } => this.match_expr(
                destination,
                block,
                scrutinee,
                arms,
                expr_span,
                this.thir[scrutinee].span,
            ),
            ExprKind::If { cond, then, else_opt, if_then_scope } => {
                let then_span = this.thir[then].span;
                let then_source_info = this.source_info(then_span);
                let condition_scope = this.local_scope();

                let then_and_else_blocks = this.in_scope(
                    (if_then_scope, then_source_info),
                    LintLevel::Inherited,
                    |this| {
                        // FIXME: Does this need extra logic to handle let-chains?
                        let source_info = if this.is_let(cond) {
                            let variable_scope =
                                this.new_source_scope(then_span, LintLevel::Inherited);
                            this.source_scope = variable_scope;
                            SourceInfo { span: then_span, scope: variable_scope }
                        } else {
                            this.source_info(then_span)
                        };

                        // Lower the condition, and have it branch into `then` and `else` blocks.
                        let (then_block, else_block) =
                            this.in_if_then_scope(condition_scope, then_span, |this| {
                                let then_blk = this
                                    .then_else_break(
                                        block,
                                        cond,
                                        Some(condition_scope), // Temp scope
                                        source_info,
                                        DeclareLetBindings::Yes, // Declare `let` bindings normally
                                    )
                                    .into_block();

                                // Lower the `then` arm into its block.
                                this.expr_into_dest(destination, then_blk, then)
                            });

                        // Pack `(then_block, else_block)` into `BlockAnd<BasicBlock>`.
                        then_block.and(else_block)
                    },
                );

                // Unpack `BlockAnd<BasicBlock>` into `(then_blk, else_blk)`.
                let (then_blk, mut else_blk);
                else_blk = unpack!(then_blk = then_and_else_blocks);

                // If there is an `else` arm, lower it into `else_blk`.
                if let Some(else_expr) = else_opt {
                    else_blk = this.expr_into_dest(destination, else_blk, else_expr).into_block();
                } else {
                    // There is no `else` arm, so we know both arms have type `()`.
                    // Generate the implicit `else {}` by assigning unit.
                    let correct_si = this.source_info(expr_span.shrink_to_hi());
                    this.cfg.push_assign_unit(else_blk, correct_si, destination, this.tcx);
                }

                // The `then` and `else` arms have been lowered into their respective
                // blocks, so make both of them meet up in a new block.
                let join_block = this.cfg.start_new_block();
                this.cfg.goto(then_blk, source_info, join_block);
                this.cfg.goto(else_blk, source_info, join_block);
                join_block.unit()
            }
            ExprKind::Let { .. } => {
                // After desugaring, `let` expressions should only appear inside `if`
                // expressions or `match` guards, possibly nested within a let-chain.
                // In both cases they are specifically handled by the lowerings of
                // those expressions, so this case is currently unreachable.
                span_bug!(expr_span, "unexpected let expression outside of if or match-guard");
            }
            ExprKind::NeverToAny { source } => {
                let source_expr = &this.thir[source];
                let is_call =
                    matches!(source_expr.kind, ExprKind::Call { .. } | ExprKind::InlineAsm { .. });

                // (#66975) Source could be a const of type `!`, so has to
                // exist in the generated MIR.
                unpack!(
                    block =
                        this.as_temp(block, this.local_temp_lifetime(), source, Mutability::Mut)
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
                let condition_scope = this.local_scope();
                let source_info = this.source_info(expr.span);

                this.visit_coverage_branch_operation(op, expr.span);

                // We first evaluate the left-hand side of the predicate ...
                let (then_block, else_block) =
                    this.in_if_then_scope(condition_scope, expr.span, |this| {
                        this.then_else_break(
                            block,
                            lhs,
                            Some(condition_scope), // Temp scope
                            source_info,
                            // This flag controls how inner `let` expressions are lowered,
                            // but either way there shouldn't be any of those in here.
                            DeclareLetBindings::LetNotPermitted,
                        )
                    });
                let (short_circuit, continuation, constant) = match op {
                    LogicalOp::And => (else_block, then_block, false),
                    LogicalOp::Or => (then_block, else_block, true),
                };
                // At this point, the control flow splits into a short-circuiting path
                // and a continuation path.
                // - If the operator is `&&`, passing `lhs` leads to continuation of evaluation on `rhs`;
                //   failing it leads to the short-circuting path which assigns `false` to the place.
                // - If the operator is `||`, failing `lhs` leads to continuation of evaluation on `rhs`;
                //   passing it leads to the short-circuting path which assigns `true` to the place.
                this.cfg.push_assign_constant(
                    short_circuit,
                    source_info,
                    destination,
                    ConstOperand {
                        span: expr.span,
                        user_ty: None,
                        const_: Const::from_bool(this.tcx, constant),
                    },
                );
                let mut rhs_block =
                    this.expr_into_dest(destination, continuation, rhs).into_block();
                // Instrument the lowered RHS's value for condition coverage.
                // (Does nothing if condition coverage is not enabled.)
                this.visit_coverage_standalone_condition(rhs, destination, &mut rhs_block);

                let target = this.cfg.start_new_block();
                this.cfg.goto(rhs_block, source_info, target);
                this.cfg.goto(short_circuit, source_info, target);
                target.unit()
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
                        TerminatorKind::FalseUnwind {
                            real_target: body_block,
                            unwind: UnwindAction::Continue,
                        },
                    );
                    this.diverge_from(loop_block);

                    // The “return” value of the loop body must always be a unit. We therefore
                    // introduce a unit temporary as the destination for the loop body.
                    let tmp = this.get_unit_temp();
                    // Execute the body, branching back to the test.
                    let body_block_end = this.expr_into_dest(tmp, body_block, body).into_block();
                    this.cfg.goto(body_block_end, source_info, loop_block);

                    // Loops are only exited by `break` expressions.
                    None
                })
            }
            ExprKind::Call { ty: _, fun, ref args, from_hir_call, fn_span } => {
                let fun = unpack!(block = this.as_local_operand(block, fun));
                let args: Box<[_]> = args
                    .into_iter()
                    .copied()
                    .map(|arg| Spanned {
                        node: unpack!(block = this.as_local_call_operand(block, arg)),
                        span: this.thir.exprs[arg].span,
                    })
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
                        unwind: UnwindAction::Continue,
                        destination,
                        // The presence or absence of a return edge affects control-flow sensitive
                        // MIR checks and ultimately whether code is accepted or not. We can only
                        // omit the return edge if a return type is visibly uninhabited to a module
                        // that makes the call.
                        target: expr
                            .ty
                            .is_inhabited_from(
                                this.tcx,
                                this.parent_module,
                                this.infcx.typing_env(this.param_env),
                            )
                            .then_some(success),
                        call_source: if from_hir_call {
                            CallSource::Normal
                        } else {
                            CallSource::OverloadedOperator
                        },
                        fn_span,
                    },
                );
                this.diverge_from(block);
                success.unit()
            }
            ExprKind::ByUse { expr, span } => {
                let place = unpack!(block = this.as_place(block, expr));
                let ty = place.ty(&this.local_decls, this.tcx).ty;

                if this.tcx.type_is_copy_modulo_regions(this.infcx.typing_env(this.param_env), ty) {
                    this.cfg.push_assign(
                        block,
                        source_info,
                        destination,
                        Rvalue::Use(Operand::Copy(place)),
                    );
                    block.unit()
                } else if this.infcx.type_is_use_cloned_modulo_regions(this.param_env, ty) {
                    // Convert `expr.use` to a call like `Clone::clone(&expr)`
                    let success = this.cfg.start_new_block();
                    let clone_trait = this.tcx.require_lang_item(LangItem::Clone, span);
                    let clone_fn = this.tcx.associated_item_def_ids(clone_trait)[0];
                    let func = Operand::function_handle(this.tcx, clone_fn, [ty.into()], expr_span);
                    let ref_ty = Ty::new_imm_ref(this.tcx, this.tcx.lifetimes.re_erased, ty);
                    let ref_place = this.temp(ref_ty, span);
                    this.cfg.push_assign(
                        block,
                        source_info,
                        ref_place,
                        Rvalue::Ref(this.tcx.lifetimes.re_erased, BorrowKind::Shared, place),
                    );
                    this.cfg.terminate(
                        block,
                        source_info,
                        TerminatorKind::Call {
                            func,
                            args: [Spanned { node: Operand::Move(ref_place), span: DUMMY_SP }]
                                .into(),
                            destination,
                            target: Some(success),
                            unwind: UnwindAction::Unreachable,
                            call_source: CallSource::Use,
                            fn_span: expr_span,
                        },
                    );
                    success.unit()
                } else {
                    this.cfg.push_assign(
                        block,
                        source_info,
                        destination,
                        Rvalue::Use(Operand::Move(place)),
                    );
                    block.unit()
                }
            }
            ExprKind::Use { source } => this.expr_into_dest(destination, block, source),
            ExprKind::Borrow { arg, borrow_kind } => {
                // We don't do this in `as_rvalue` because we use `as_place`
                // for borrow expressions, so we cannot create an `RValue` that
                // remains valid across user code. `as_rvalue` is usually called
                // by this method anyway, so this shouldn't cause too many
                // unnecessary temporaries.
                let arg_place = match borrow_kind {
                    BorrowKind::Shared => {
                        unpack!(block = this.as_read_only_place(block, arg))
                    }
                    _ => unpack!(block = this.as_place(block, arg)),
                };
                let borrow = Rvalue::Ref(this.tcx.lifetimes.re_erased, borrow_kind, arg_place);
                this.cfg.push_assign(block, source_info, destination, borrow);
                block.unit()
            }
            ExprKind::RawBorrow { mutability, arg } => {
                let place = match mutability {
                    hir::Mutability::Not => this.as_read_only_place(block, arg),
                    hir::Mutability::Mut => this.as_place(block, arg),
                };
                let address_of = Rvalue::RawPtr(mutability.into(), unpack!(block = place));
                this.cfg.push_assign(block, source_info, destination, address_of);
                block.unit()
            }
            ExprKind::Adt(box AdtExpr {
                adt_def,
                variant_index,
                args,
                ref user_ty,
                ref fields,
                ref base,
            }) => {
                // See the notes for `ExprKind::Array` in `as_rvalue` and for
                // `ExprKind::Borrow` above.
                let is_union = adt_def.is_union();
                let active_field_index = is_union.then(|| fields[0].name);

                let scope = this.local_temp_lifetime();

                // first process the set of fields that were provided
                // (evaluating them in order given by user)
                let fields_map: FxHashMap<_, _> = fields
                    .into_iter()
                    .map(|f| {
                        (
                            f.name,
                            unpack!(
                                block = this.as_operand(
                                    block,
                                    scope,
                                    f.expr,
                                    LocalInfo::AggregateTemp,
                                    NeedsTemporary::Maybe,
                                )
                            ),
                        )
                    })
                    .collect();

                let variant = adt_def.variant(variant_index);
                let field_names = variant.fields.indices();

                let fields = match base {
                    AdtExprBase::None => {
                        field_names.filter_map(|n| fields_map.get(&n).cloned()).collect()
                    }
                    AdtExprBase::Base(FruInfo { base, field_types }) => {
                        let place_builder = unpack!(block = this.as_place_builder(block, *base));

                        // We desugar FRU as we lower to MIR, so for each
                        // base-supplied field, generate an operand that
                        // reads it from the base.
                        itertools::zip_eq(field_names, &**field_types)
                            .map(|(n, ty)| match fields_map.get(&n) {
                                Some(v) => v.clone(),
                                None => {
                                    let place =
                                        place_builder.clone_project(PlaceElem::Field(n, *ty));
                                    this.consume_by_copy_or_move(place.to_place(this))
                                }
                            })
                            .collect()
                    }
                    AdtExprBase::DefaultFields(field_types) => {
                        itertools::zip_eq(field_names, field_types)
                            .map(|(n, &ty)| match fields_map.get(&n) {
                                Some(v) => v.clone(),
                                None => match variant.fields[n].value {
                                    Some(def) => {
                                        let value = Const::Unevaluated(
                                            UnevaluatedConst::new(def, args),
                                            ty,
                                        );
                                        Operand::Constant(Box::new(ConstOperand {
                                            span: expr_span,
                                            user_ty: None,
                                            const_: value,
                                        }))
                                    }
                                    None => {
                                        let name = variant.fields[n].name;
                                        span_bug!(
                                            expr_span,
                                            "missing mandatory field `{name}` of type `{ty}`",
                                        );
                                    }
                                },
                            })
                            .collect()
                    }
                };

                let inferred_ty = expr.ty;
                let user_ty = user_ty.as_ref().map(|user_ty| {
                    this.canonical_user_type_annotations.push(CanonicalUserTypeAnnotation {
                        span: source_info.span,
                        user_ty: user_ty.clone(),
                        inferred_ty,
                    })
                });
                let adt = Box::new(AggregateKind::Adt(
                    adt_def.did(),
                    variant_index,
                    args,
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
            ExprKind::InlineAsm(box InlineAsmExpr {
                asm_macro,
                template,
                ref operands,
                options,
                line_spans,
            }) => {
                use rustc_middle::{mir, thir};

                let destination_block = this.cfg.start_new_block();
                let mut targets =
                    if asm_macro.diverges(options) { vec![] } else { vec![destination_block] };

                let operands = operands
                    .into_iter()
                    .map(|op| match *op {
                        thir::InlineAsmOperand::In { reg, expr } => mir::InlineAsmOperand::In {
                            reg,
                            value: unpack!(block = this.as_local_operand(block, expr)),
                        },
                        thir::InlineAsmOperand::Out { reg, late, expr } => {
                            mir::InlineAsmOperand::Out {
                                reg,
                                late,
                                place: expr.map(|expr| unpack!(block = this.as_place(block, expr))),
                            }
                        }
                        thir::InlineAsmOperand::InOut { reg, late, expr } => {
                            let place = unpack!(block = this.as_place(block, expr));
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
                                in_value: unpack!(block = this.as_local_operand(block, in_expr)),
                                out_place: out_expr.map(|out_expr| {
                                    unpack!(block = this.as_place(block, out_expr))
                                }),
                            }
                        }
                        thir::InlineAsmOperand::Const { value, span } => {
                            mir::InlineAsmOperand::Const {
                                value: Box::new(ConstOperand {
                                    span,
                                    user_ty: None,
                                    const_: value,
                                }),
                            }
                        }
                        thir::InlineAsmOperand::SymFn { value } => mir::InlineAsmOperand::SymFn {
                            value: Box::new(this.as_constant(&this.thir[value])),
                        },
                        thir::InlineAsmOperand::SymStatic { def_id } => {
                            mir::InlineAsmOperand::SymStatic { def_id }
                        }
                        thir::InlineAsmOperand::Label { block } => {
                            let target = this.cfg.start_new_block();
                            let target_index = targets.len();
                            targets.push(target);

                            let tmp = this.get_unit_temp();
                            let target =
                                this.ast_block(tmp, target, block, source_info).into_block();
                            this.cfg.terminate(
                                target,
                                source_info,
                                TerminatorKind::Goto { target: destination_block },
                            );

                            mir::InlineAsmOperand::Label { target_index }
                        }
                    })
                    .collect();

                if !expr.ty.is_never() {
                    this.cfg.push_assign_unit(block, source_info, destination, this.tcx);
                }

                let asm_macro = match asm_macro {
                    AsmMacro::Asm | AsmMacro::GlobalAsm => InlineAsmMacro::Asm,
                    AsmMacro::NakedAsm => InlineAsmMacro::NakedAsm,
                };

                this.cfg.terminate(
                    block,
                    source_info,
                    TerminatorKind::InlineAsm {
                        asm_macro,
                        template,
                        operands,
                        options,
                        line_spans,
                        targets: targets.into_boxed_slice(),
                        unwind: if options.contains(InlineAsmOptions::MAY_UNWIND) {
                            UnwindAction::Continue
                        } else {
                            UnwindAction::Unreachable
                        },
                    },
                );
                if options.contains(InlineAsmOptions::MAY_UNWIND) {
                    this.diverge_from(block);
                }
                destination_block.unit()
            }

            // These cases don't actually need a destination
            ExprKind::Assign { .. } | ExprKind::AssignOp { .. } => {
                block = this.stmt_expr(block, expr_id, None).into_block();
                this.cfg.push_assign_unit(block, source_info, destination, this.tcx);
                block.unit()
            }

            ExprKind::Continue { .. }
            | ExprKind::Break { .. }
            | ExprKind::Return { .. }
            | ExprKind::Become { .. } => {
                block = this.stmt_expr(block, expr_id, None).into_block();
                // No assign, as these have type `!`.
                block.unit()
            }

            // Avoid creating a temporary
            ExprKind::VarRef { .. }
            | ExprKind::UpvarRef { .. }
            | ExprKind::PlaceTypeAscription { .. }
            | ExprKind::ValueTypeAscription { .. }
            | ExprKind::PlaceUnwrapUnsafeBinder { .. }
            | ExprKind::ValueUnwrapUnsafeBinder { .. } => {
                debug_assert!(Category::of(&expr.kind) == Some(Category::Place));

                let place = unpack!(block = this.as_place(block, expr_id));
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

                let place = unpack!(block = this.as_place(block, expr_id));
                let rvalue = Rvalue::Use(this.consume_by_copy_or_move(place));
                this.cfg.push_assign(block, source_info, destination, rvalue);
                block.unit()
            }

            ExprKind::Yield { value } => {
                let scope = this.local_temp_lifetime();
                let value = unpack!(
                    block =
                        this.as_operand(block, scope, value, LocalInfo::Boring, NeedsTemporary::No)
                );
                let resume = this.cfg.start_new_block();
                this.cfg.terminate(
                    block,
                    source_info,
                    TerminatorKind::Yield { value, resume, resume_arg: destination, drop: None },
                );
                this.coroutine_drop_cleanup(block);
                resume.unit()
            }

            // these are the cases that are more naturally handled by some other mode
            ExprKind::Unary { .. }
            | ExprKind::Binary { .. }
            | ExprKind::Box { .. }
            | ExprKind::Cast { .. }
            | ExprKind::PointerCoercion { .. }
            | ExprKind::Repeat { .. }
            | ExprKind::Array { .. }
            | ExprKind::Tuple { .. }
            | ExprKind::Closure { .. }
            | ExprKind::ConstBlock { .. }
            | ExprKind::Literal { .. }
            | ExprKind::NamedConst { .. }
            | ExprKind::NonHirLiteral { .. }
            | ExprKind::ZstLiteral { .. }
            | ExprKind::ConstParam { .. }
            | ExprKind::ThreadLocalRef(_)
            | ExprKind::StaticRef { .. }
            | ExprKind::OffsetOf { .. }
            | ExprKind::WrapUnsafeBinder { .. } => {
                debug_assert!(match Category::of(&expr.kind).unwrap() {
                    // should be handled above
                    Category::Rvalue(RvalueFunc::Into) => false,

                    // must be handled above or else we get an
                    // infinite loop in the builder; see
                    // e.g., `ExprKind::VarRef` above
                    Category::Place => false,

                    _ => true,
                });

                let rvalue = unpack!(block = this.as_local_rvalue(block, expr_id));
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

    fn is_let(&self, expr: ExprId) -> bool {
        match self.thir[expr].kind {
            ExprKind::Let { .. } => true,
            ExprKind::Scope { value, .. } => self.is_let(value),
            _ => false,
        }
    }
}
