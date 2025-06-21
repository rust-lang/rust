use rustc_abi::{Scalar, Size, TagEncoding, Variants, WrappingRange};
use rustc_hir::LangItem;
use rustc_index::IndexVec;
use rustc_middle::bug;
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::*;
use rustc_middle::ty::layout::PrimitiveExt;
use rustc_middle::ty::{self, Ty, TyCtxt, TypingEnv};
use rustc_session::Session;
use tracing::debug;

/// This pass inserts checks for a valid enum discriminant where they are most
/// likely to find UB, because checking everywhere like Miri would generate too
/// much MIR.
pub(super) struct CheckEnums;

impl<'tcx> crate::MirPass<'tcx> for CheckEnums {
    fn is_enabled(&self, sess: &Session) -> bool {
        sess.ub_checks()
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        // This pass emits new panics. If for whatever reason we do not have a panic
        // implementation, running this pass may cause otherwise-valid code to not compile.
        if tcx.lang_items().get(LangItem::PanicImpl).is_none() {
            return;
        }

        let typing_env = body.typing_env(tcx);
        let basic_blocks = body.basic_blocks.as_mut();
        let local_decls = &mut body.local_decls;

        // This operation inserts new blocks. Each insertion changes the Location for all
        // statements/blocks after. Iterating or visiting the MIR in order would require updating
        // our current location after every insertion. By iterating backwards, we dodge this issue:
        // The only Locations that an insertion changes have already been handled.
        for block in basic_blocks.indices().rev() {
            for statement_index in (0..basic_blocks[block].statements.len()).rev() {
                let location = Location { block, statement_index };
                let statement = &basic_blocks[block].statements[statement_index];
                let source_info = statement.source_info;

                let mut finder = EnumFinder::new(tcx, local_decls, typing_env);
                finder.visit_statement(statement, location);

                for check in finder.into_found_enums() {
                    debug!("Inserting enum check");
                    let new_block = split_block(basic_blocks, location);

                    match check {
                        EnumCheckType::Direct { source_op, discr, op_size, valid_discrs } => {
                            insert_direct_enum_check(
                                tcx,
                                local_decls,
                                basic_blocks,
                                block,
                                source_op,
                                discr,
                                op_size,
                                valid_discrs,
                                source_info,
                                new_block,
                            )
                        }
                        EnumCheckType::Uninhabited => insert_uninhabited_enum_check(
                            tcx,
                            local_decls,
                            &mut basic_blocks[block],
                            source_info,
                            new_block,
                        ),
                        EnumCheckType::WithNiche {
                            source_op,
                            discr,
                            op_size,
                            offset,
                            valid_range,
                        } => insert_niche_check(
                            tcx,
                            local_decls,
                            &mut basic_blocks[block],
                            source_op,
                            valid_range,
                            discr,
                            op_size,
                            offset,
                            source_info,
                            new_block,
                        ),
                    }
                }
            }
        }
    }

    fn is_required(&self) -> bool {
        true
    }
}

/// Represent the different kind of enum checks we can insert.
enum EnumCheckType<'tcx> {
    /// We know we try to create an uninhabited enum from an inhabited variant.
    Uninhabited,
    /// We know the enum does no niche optimizations and can thus easily compute
    /// the valid discriminants.
    Direct {
        source_op: Operand<'tcx>,
        discr: TyAndSize<'tcx>,
        op_size: Size,
        valid_discrs: Vec<u128>,
    },
    /// We try to construct an enum that has a niche.
    WithNiche {
        source_op: Operand<'tcx>,
        discr: TyAndSize<'tcx>,
        op_size: Size,
        offset: Size,
        valid_range: WrappingRange,
    },
}

struct TyAndSize<'tcx> {
    pub ty: Ty<'tcx>,
    pub size: Size,
}

/// A [Visitor] that finds the construction of enums and evaluates which checks
/// we should apply.
struct EnumFinder<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    local_decls: &'a mut LocalDecls<'tcx>,
    typing_env: TypingEnv<'tcx>,
    enums: Vec<EnumCheckType<'tcx>>,
}

impl<'a, 'tcx> EnumFinder<'a, 'tcx> {
    fn new(
        tcx: TyCtxt<'tcx>,
        local_decls: &'a mut LocalDecls<'tcx>,
        typing_env: TypingEnv<'tcx>,
    ) -> Self {
        EnumFinder { tcx, local_decls, typing_env, enums: Vec::new() }
    }

    /// Returns the found enum creations and which checks should be inserted.
    fn into_found_enums(self) -> Vec<EnumCheckType<'tcx>> {
        self.enums
    }
}

impl<'a, 'tcx> Visitor<'tcx> for EnumFinder<'a, 'tcx> {
    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        if let Rvalue::Cast(CastKind::Transmute, op, ty) = rvalue {
            let ty::Adt(adt_def, _) = ty.kind() else {
                return;
            };
            if !adt_def.is_enum() {
                return;
            }

            let Ok(enum_layout) = self.tcx.layout_of(self.typing_env.as_query_input(*ty)) else {
                return;
            };
            let Ok(op_layout) = self
                .tcx
                .layout_of(self.typing_env.as_query_input(op.ty(self.local_decls, self.tcx)))
            else {
                return;
            };

            match enum_layout.variants {
                Variants::Empty if op_layout.is_uninhabited() => return,
                // An empty enum that tries to be constructed from an inhabited value, this
                // is never correct.
                Variants::Empty => {
                    // The enum layout is uninhabited but we construct it from sth inhabited.
                    // This is always UB.
                    self.enums.push(EnumCheckType::Uninhabited);
                }
                // Construction of Single value enums is always fine.
                Variants::Single { .. } => {}
                // Construction of an enum with multiple variants but no niche optimizations.
                Variants::Multiple {
                    tag_encoding: TagEncoding::Direct,
                    tag: Scalar::Initialized { value, .. },
                    ..
                } => {
                    let valid_discrs =
                        adt_def.discriminants(self.tcx).map(|(_, discr)| discr.val).collect();

                    let discr =
                        TyAndSize { ty: value.to_int_ty(self.tcx), size: value.size(&self.tcx) };
                    self.enums.push(EnumCheckType::Direct {
                        source_op: op.to_copy(),
                        discr,
                        op_size: op_layout.size,
                        valid_discrs,
                    });
                }
                // Construction of an enum with multiple variants and niche optimizations.
                Variants::Multiple {
                    tag_encoding: TagEncoding::Niche { .. },
                    tag: Scalar::Initialized { value, valid_range, .. },
                    tag_field,
                    ..
                } => {
                    let discr =
                        TyAndSize { ty: value.to_int_ty(self.tcx), size: value.size(&self.tcx) };
                    self.enums.push(EnumCheckType::WithNiche {
                        source_op: op.to_copy(),
                        discr,
                        op_size: op_layout.size,
                        offset: enum_layout.fields.offset(tag_field.as_usize()),
                        valid_range,
                    });
                }
                _ => return,
            }

            self.super_rvalue(rvalue, location);
        }
    }
}

fn split_block(
    basic_blocks: &mut IndexVec<BasicBlock, BasicBlockData<'_>>,
    location: Location,
) -> BasicBlock {
    let block_data = &mut basic_blocks[location.block];

    // Drain every statement after this one and move the current terminator to a new basic block.
    let new_block = BasicBlockData {
        statements: block_data.statements.split_off(location.statement_index),
        terminator: block_data.terminator.take(),
        is_cleanup: block_data.is_cleanup,
    };

    basic_blocks.push(new_block)
}

/// Inserts the cast of an operand (any type) to a u128 value that holds the discriminant value.
fn insert_discr_cast_to_u128<'tcx>(
    tcx: TyCtxt<'tcx>,
    local_decls: &mut IndexVec<Local, LocalDecl<'tcx>>,
    block_data: &mut BasicBlockData<'tcx>,
    source_op: Operand<'tcx>,
    discr: TyAndSize<'tcx>,
    op_size: Size,
    offset: Option<Size>,
    source_info: SourceInfo,
) -> Place<'tcx> {
    let get_ty_for_size = |tcx: TyCtxt<'tcx>, size: Size| -> Ty<'tcx> {
        match size.bytes() {
            1 => tcx.types.u8,
            2 => tcx.types.u16,
            4 => tcx.types.u32,
            8 => tcx.types.u64,
            16 => tcx.types.u128,
            invalid => bug!("Found discriminant with invalid size, has {} bytes", invalid),
        }
    };

    let (cast_kind, discr_ty_bits) = if discr.size.bytes() < op_size.bytes() {
        // The discriminant is less wide than the operand, cast the operand into
        // [MaybeUninit; N] and then index into it.
        let mu = Ty::new_maybe_uninit(tcx, tcx.types.u8);
        let array_len = op_size.bytes();
        let mu_array_ty = Ty::new_array(tcx, mu, array_len);
        let mu_array =
            local_decls.push(LocalDecl::with_source_info(mu_array_ty, source_info)).into();
        let rvalue = Rvalue::Cast(CastKind::Transmute, source_op, mu_array_ty);
        block_data.statements.push(Statement {
            source_info,
            kind: StatementKind::Assign(Box::new((mu_array, rvalue))),
        });

        // Index into the array of MaybeUninit to get something that is actually
        // as wide as the discriminant.
        let offset = offset.unwrap_or(Size::ZERO);
        let smaller_mu_array = mu_array.project_deeper(
            &[ProjectionElem::Subslice {
                from: offset.bytes(),
                to: offset.bytes() + discr.size.bytes(),
                from_end: false,
            }],
            tcx,
        );

        (CastKind::Transmute, Operand::Copy(smaller_mu_array))
    } else {
        let operand_int_ty = get_ty_for_size(tcx, op_size);

        let op_as_int =
            local_decls.push(LocalDecl::with_source_info(operand_int_ty, source_info)).into();
        let rvalue = Rvalue::Cast(CastKind::Transmute, source_op, operand_int_ty);
        block_data.statements.push(Statement {
            source_info,
            kind: StatementKind::Assign(Box::new((op_as_int, rvalue))),
        });

        (CastKind::IntToInt, Operand::Copy(op_as_int))
    };

    // Cast the resulting value to the actual discriminant integer type.
    let rvalue = Rvalue::Cast(cast_kind, discr_ty_bits, discr.ty);
    let discr_in_discr_ty =
        local_decls.push(LocalDecl::with_source_info(discr.ty, source_info)).into();
    block_data.statements.push(Statement {
        source_info,
        kind: StatementKind::Assign(Box::new((discr_in_discr_ty, rvalue))),
    });

    // Cast the discriminant to a u128 (base for comparisions of enum discriminants).
    let const_u128 = Ty::new_uint(tcx, ty::UintTy::U128);
    let rvalue = Rvalue::Cast(CastKind::IntToInt, Operand::Copy(discr_in_discr_ty), const_u128);
    let discr = local_decls.push(LocalDecl::with_source_info(const_u128, source_info)).into();
    block_data
        .statements
        .push(Statement { source_info, kind: StatementKind::Assign(Box::new((discr, rvalue))) });

    discr
}

fn insert_direct_enum_check<'tcx>(
    tcx: TyCtxt<'tcx>,
    local_decls: &mut IndexVec<Local, LocalDecl<'tcx>>,
    basic_blocks: &mut IndexVec<BasicBlock, BasicBlockData<'tcx>>,
    current_block: BasicBlock,
    source_op: Operand<'tcx>,
    discr: TyAndSize<'tcx>,
    op_size: Size,
    discriminants: Vec<u128>,
    source_info: SourceInfo,
    new_block: BasicBlock,
) {
    // Insert a new target block that is branched to in case of an invalid discriminant.
    let invalid_discr_block_data = BasicBlockData::new(None, false);
    let invalid_discr_block = basic_blocks.push(invalid_discr_block_data);
    let block_data = &mut basic_blocks[current_block];
    let discr = insert_discr_cast_to_u128(
        tcx,
        local_decls,
        block_data,
        source_op,
        discr,
        op_size,
        None,
        source_info,
    );

    // Branch based on the discriminant value.
    block_data.terminator = Some(Terminator {
        source_info,
        kind: TerminatorKind::SwitchInt {
            discr: Operand::Copy(discr),
            targets: SwitchTargets::new(
                discriminants.into_iter().map(|discr| (discr, new_block)),
                invalid_discr_block,
            ),
        },
    });

    // Abort in case of an invalid enum discriminant.
    basic_blocks[invalid_discr_block].terminator = Some(Terminator {
        source_info,
        kind: TerminatorKind::Assert {
            cond: Operand::Constant(Box::new(ConstOperand {
                span: source_info.span,
                user_ty: None,
                const_: Const::Val(ConstValue::from_bool(false), tcx.types.bool),
            })),
            expected: true,
            target: new_block,
            msg: Box::new(AssertKind::InvalidEnumConstruction(Operand::Copy(discr))),
            // This calls panic_invalid_enum_construction, which is #[rustc_nounwind].
            // We never want to insert an unwind into unsafe code, because unwinding could
            // make a failing UB check turn into much worse UB when we start unwinding.
            unwind: UnwindAction::Unreachable,
        },
    });
}

fn insert_uninhabited_enum_check<'tcx>(
    tcx: TyCtxt<'tcx>,
    local_decls: &mut IndexVec<Local, LocalDecl<'tcx>>,
    block_data: &mut BasicBlockData<'tcx>,
    source_info: SourceInfo,
    new_block: BasicBlock,
) {
    let is_ok: Place<'_> =
        local_decls.push(LocalDecl::with_source_info(tcx.types.bool, source_info)).into();
    block_data.statements.push(Statement {
        source_info,
        kind: StatementKind::Assign(Box::new((
            is_ok,
            Rvalue::Use(Operand::Constant(Box::new(ConstOperand {
                span: source_info.span,
                user_ty: None,
                const_: Const::Val(ConstValue::from_bool(false), tcx.types.bool),
            }))),
        ))),
    });

    block_data.terminator = Some(Terminator {
        source_info,
        kind: TerminatorKind::Assert {
            cond: Operand::Copy(is_ok),
            expected: true,
            target: new_block,
            msg: Box::new(AssertKind::InvalidEnumConstruction(Operand::Constant(Box::new(
                ConstOperand {
                    span: source_info.span,
                    user_ty: None,
                    const_: Const::Val(ConstValue::from_u128(0), tcx.types.u128),
                },
            )))),
            // This calls panic_invalid_enum_construction, which is #[rustc_nounwind].
            // We never want to insert an unwind into unsafe code, because unwinding could
            // make a failing UB check turn into much worse UB when we start unwinding.
            unwind: UnwindAction::Unreachable,
        },
    });
}

fn insert_niche_check<'tcx>(
    tcx: TyCtxt<'tcx>,
    local_decls: &mut IndexVec<Local, LocalDecl<'tcx>>,
    block_data: &mut BasicBlockData<'tcx>,
    source_op: Operand<'tcx>,
    valid_range: WrappingRange,
    discr: TyAndSize<'tcx>,
    op_size: Size,
    offset: Size,
    source_info: SourceInfo,
    new_block: BasicBlock,
) {
    let discr = insert_discr_cast_to_u128(
        tcx,
        local_decls,
        block_data,
        source_op,
        discr,
        op_size,
        Some(offset),
        source_info,
    );

    // Compare the discriminant agains the valid_range.
    let start_const = Operand::Constant(Box::new(ConstOperand {
        span: source_info.span,
        user_ty: None,
        const_: Const::Val(ConstValue::from_u128(valid_range.start), tcx.types.u128),
    }));
    let end_start_diff_const = Operand::Constant(Box::new(ConstOperand {
        span: source_info.span,
        user_ty: None,
        const_: Const::Val(
            ConstValue::from_u128(u128::wrapping_sub(valid_range.end, valid_range.start)),
            tcx.types.u128,
        ),
    }));

    let discr_diff: Place<'_> =
        local_decls.push(LocalDecl::with_source_info(tcx.types.u128, source_info)).into();
    block_data.statements.push(Statement {
        source_info,
        kind: StatementKind::Assign(Box::new((
            discr_diff,
            Rvalue::BinaryOp(BinOp::Sub, Box::new((Operand::Copy(discr), start_const))),
        ))),
    });

    let is_ok: Place<'_> =
        local_decls.push(LocalDecl::with_source_info(tcx.types.bool, source_info)).into();
    block_data.statements.push(Statement {
        source_info,
        kind: StatementKind::Assign(Box::new((
            is_ok,
            Rvalue::BinaryOp(
                // This is a `WrappingRange`, so make sure to get the wrapping right.
                BinOp::Le,
                Box::new((Operand::Copy(discr_diff), end_start_diff_const)),
            ),
        ))),
    });

    block_data.terminator = Some(Terminator {
        source_info,
        kind: TerminatorKind::Assert {
            cond: Operand::Copy(is_ok),
            expected: true,
            target: new_block,
            msg: Box::new(AssertKind::InvalidEnumConstruction(Operand::Copy(discr))),
            // This calls panic_invalid_enum_construction, which is #[rustc_nounwind].
            // We never want to insert an unwind into unsafe code, because unwinding could
            // make a failing UB check turn into much worse UB when we start unwinding.
            unwind: UnwindAction::Unreachable,
        },
    });
}
