use crate::MirPass;
use rustc_hir::lang_items::LangItem;
use rustc_index::IndexVec;
use rustc_middle::mir::interpret::Scalar;
use rustc_middle::mir::visit::NonMutatingUseContext;
use rustc_middle::mir::visit::PlaceContext;
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::*;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{ParamEnv, ParamEnvAnd, Ty, TyCtxt};
use rustc_session::Session;
use rustc_target::abi::{Integer, Niche, Primitive, Size};

pub struct CheckNiches;

impl<'tcx> MirPass<'tcx> for CheckNiches {
    fn is_enabled(&self, sess: &Session) -> bool {
        sess.opts.debug_assertions
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        // This pass emits new panics. If for whatever reason we do not have a panic
        // implementation, running this pass may cause otherwise-valid code to not compile.
        if tcx.lang_items().get(LangItem::PanicImpl).is_none() {
            return;
        }

        let is_generator = tcx.type_of(body.source.def_id()).instantiate_identity().is_generator();
        if is_generator {
            return;
        }

        with_no_trimmed_paths!(debug!("Inserting niche checks for {:?}", body.source));

        let basic_blocks = body.basic_blocks.as_mut();
        let param_env = tcx.param_env_reveal_all_normalized(body.source.instance.def_id());
        let local_decls: &mut IndexVec<_, _> = &mut body.local_decls;

        // This pass inserts new blocks. Each insertion changes the Location for all
        // statements/blocks after. Iterating or visiting the MIR in order would require updating
        // our current location after every insertion. By iterating backwards, we dodge this issue:
        // The only Locations that an insertion changes have already been handled.
        for block in (0..basic_blocks.len()).rev() {
            let block = block.into();
            for statement_index in (0..basic_blocks[block].statements.len()).rev() {
                let location = Location { block, statement_index };
                let statement = &basic_blocks[block].statements[statement_index];
                let source_info = statement.source_info;

                let mut finder = NicheFinder { tcx, local_decls, param_env, places: Vec::new() };
                finder.visit_statement(statement, location);

                for (place, ty, niche) in core::mem::take(&mut finder.places) {
                    with_no_trimmed_paths!(debug!("Inserting niche check for {:?}", ty));
                    let (block_data, new_block) = split_block(basic_blocks, location);
                    finder.insert_niche_check(block_data, new_block, place, niche, source_info);
                }
            }
        }
    }
}

struct NicheFinder<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    local_decls: &'a mut IndexVec<Local, LocalDecl<'tcx>>,
    param_env: ParamEnv<'tcx>,
    places: Vec<(Operand<'tcx>, Ty<'tcx>, NicheKind)>,
}

impl<'a, 'tcx> Visitor<'tcx> for NicheFinder<'a, 'tcx> {
    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        if let Rvalue::Cast(CastKind::Transmute, op, ty) = rvalue {
            if let Some(niche) = self.get_niche(*ty) {
                self.places.push((op.clone(), *ty, niche));
            }
        }

        self.super_rvalue(rvalue, location);
    }

    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
        match context {
            PlaceContext::NonMutatingUse(
                NonMutatingUseContext::Inspect
                | NonMutatingUseContext::Copy
                | NonMutatingUseContext::Move,
            ) => {}
            _ => {
                return;
            }
        }

        let ty = place.ty(self.local_decls, self.tcx).ty;
        let Some(niche) = self.get_niche(ty) else {
            return;
        };

        if ty.is_ref() {
            // References are extremely common, checking them imposes a lot of runtime overhead.
            return;
        }

        self.places.push((Operand::Copy(*place), ty, niche));

        self.super_place(place, context, location);
    }
}

impl<'a, 'tcx> NicheFinder<'a, 'tcx> {
    fn insert_niche_check(
        &mut self,
        block_data: &mut BasicBlockData<'tcx>,
        new_block: BasicBlock,
        operand: Operand<'tcx>,
        niche: NicheKind,
        source_info: SourceInfo,
    ) {
        let type_name =
            with_no_trimmed_paths!(format!("{:?}", operand.ty(self.local_decls, self.tcx)));

        let value_in_niche = self
            .local_decls
            .push(LocalDecl::with_source_info(niche.niche().ty(self.tcx), source_info))
            .into();

        match niche {
            NicheKind::Full(niche) => {
                // The niche occupies the entire source Operand, so we can just transmute
                // directly to the niche primitive.
                let rvalue = Rvalue::Cast(CastKind::Transmute, operand, niche.ty(self.tcx));
                block_data.statements.push(Statement {
                    source_info,
                    kind: StatementKind::Assign(Box::new((value_in_niche, rvalue))),
                });
            }
            NicheKind::Partial(niche, size) => {
                let mu = Ty::new_maybe_uninit(self.tcx, self.tcx.types.u8);

                // Transmute the niche-containing type to a [MaybeUninit; N]
                let mu_array_ty = Ty::new_array(self.tcx, mu, size.bytes());
                let mu_array = self
                    .local_decls
                    .push(LocalDecl::with_source_info(mu_array_ty, source_info))
                    .into();
                let rvalue = Rvalue::Cast(CastKind::Transmute, operand, mu_array_ty);
                block_data.statements.push(Statement {
                    source_info,
                    kind: StatementKind::Assign(Box::new((mu_array, rvalue))),
                });

                // Load each byte of the [MaybeUninit<u8>; N] that is part of the niche
                let niche_as_mu: Place<'tcx> = self
                    .local_decls
                    .push(LocalDecl::with_source_info(
                        Ty::new_array(self.tcx, mu, niche.size(self.tcx).bytes()),
                        source_info,
                    ))
                    .into();

                for i in 0..niche.size(self.tcx).bytes() {
                    let lhs_index = self
                        .local_decls
                        .push(LocalDecl::with_source_info(self.tcx.types.usize, source_info));
                    let index_value = Operand::Constant(Box::new(ConstOperand {
                        span: source_info.span,
                        user_ty: None,
                        const_: Const::Val(
                            ConstValue::Scalar(Scalar::from_target_usize(i, &self.tcx)),
                            self.tcx.types.usize,
                        ),
                    }));
                    block_data.statements.push(Statement {
                        source_info,
                        kind: StatementKind::Assign(Box::new((
                            lhs_index.into(),
                            Rvalue::Use(index_value),
                        ))),
                    });

                    let rhs_index = self
                        .local_decls
                        .push(LocalDecl::with_source_info(self.tcx.types.usize, source_info));
                    let index_value = Operand::Constant(Box::new(ConstOperand {
                        span: source_info.span,
                        user_ty: None,
                        const_: Const::Val(
                            ConstValue::Scalar(Scalar::from_target_usize(
                                niche.offset.bytes() + i,
                                &self.tcx,
                            )),
                            self.tcx.types.usize,
                        ),
                    }));
                    block_data.statements.push(Statement {
                        source_info,
                        kind: StatementKind::Assign(Box::new((
                            rhs_index.into(),
                            Rvalue::Use(index_value),
                        ))),
                    });

                    let lhs =
                        niche_as_mu.project_deeper(&[ProjectionElem::Index(lhs_index)], self.tcx);
                    let rhs =
                        mu_array.project_deeper(&[ProjectionElem::Index(rhs_index)], self.tcx);
                    block_data.statements.push(Statement {
                        source_info,
                        kind: StatementKind::Assign(Box::new((
                            lhs,
                            Rvalue::Use(Operand::Copy(rhs)),
                        ))),
                    });
                }

                // Transmute the smaller array of MaybeUninit<u8> to the niche primitive
                let rvalue = Rvalue::Cast(
                    CastKind::Transmute,
                    Operand::Copy(niche_as_mu),
                    niche.ty(self.tcx),
                );
                block_data.statements.push(Statement {
                    source_info,
                    kind: StatementKind::Assign(Box::new((value_in_niche, rvalue))),
                });
            }
        }

        let is_in_range = self
            .local_decls
            .push(LocalDecl::with_source_info(self.tcx.types.bool, source_info))
            .into();

        NicheCheckBuilder {
            tcx: self.tcx,
            local_decls: self.local_decls,
            block_data,
            new_block,
            niche: niche.niche(),
            value_in_niche,
            is_in_range,
            source_info,
        }
        .insert_niche_check(type_name);
    }

    fn get_niche(&self, ty: Ty<'tcx>) -> Option<NicheKind> {
        // If we can't get the layout of this, just skip the check.
        let query = ParamEnvAnd { value: ty, param_env: self.param_env };
        let Ok(layout) = self.tcx.layout_of(query) else {
            return None;
        };

        let niche = layout.largest_niche?;

        if niche.size(self.tcx) == layout.size {
            Some(NicheKind::Full(niche))
        } else {
            Some(NicheKind::Partial(niche, layout.size))
        }
    }
}

#[derive(Clone, Copy)]
enum NicheKind {
    Full(Niche),
    // We need the full Size of the type in order to do the transmute-to-MU approach
    Partial(Niche, Size),
}

impl NicheKind {
    fn niche(self) -> Niche {
        use NicheKind::*;
        match self {
            Full(niche) => niche,
            Partial(niche, _size) => niche,
        }
    }
}

fn split_block<'a, 'tcx>(
    basic_blocks: &'a mut IndexVec<BasicBlock, BasicBlockData<'tcx>>,
    location: Location,
) -> (&'a mut BasicBlockData<'tcx>, BasicBlock) {
    let block_data = &mut basic_blocks[location.block];

    // Drain every statement after this one and move the current terminator to a new basic block
    let new_block = BasicBlockData {
        statements: block_data.statements.drain(location.statement_index..).collect(),
        terminator: block_data.terminator.take(),
        is_cleanup: block_data.is_cleanup,
    };

    let new_block = basic_blocks.push(new_block);
    let block_data = &mut basic_blocks[location.block];

    (block_data, new_block)
}

struct NicheCheckBuilder<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    block_data: &'a mut BasicBlockData<'tcx>,
    local_decls: &'a mut IndexVec<Local, LocalDecl<'tcx>>,
    new_block: BasicBlock,
    niche: Niche,
    value_in_niche: Place<'tcx>,
    is_in_range: Place<'tcx>,
    source_info: SourceInfo,
}

impl<'a, 'tcx> NicheCheckBuilder<'a, 'tcx> {
    fn insert_niche_check(&mut self, type_name: String) {
        let niche = self.niche;
        let tcx = self.tcx;

        let size = self.niche.size(self.tcx);

        if niche.valid_range.start == 0 {
            // The niche starts at 0, so we can just check if it is Le the end
            self.check_end_only();
        } else if niche.valid_range.end == (u128::MAX >> (128 - size.bits())) {
            // The niche ends at the max, so we can just check if it is Ge the start
            self.check_start_only();
        } else {
            self.general_case();
        }

        let start = Operand::Constant(Box::new(ConstOperand {
            span: self.source_info.span,
            user_ty: None,
            const_: Const::Val(
                ConstValue::Scalar(Scalar::from_uint(
                    niche.valid_range.start as u128,
                    Size::from_bits(128),
                )),
                tcx.types.u128,
            ),
        }));
        let end = Operand::Constant(Box::new(ConstOperand {
            span: self.source_info.span,
            user_ty: None,
            const_: Const::Val(
                ConstValue::Scalar(Scalar::from_uint(
                    niche.valid_range.end as u128,
                    Size::from_bits(128),
                )),
                tcx.types.u128,
            ),
        }));
        let offset = Operand::Constant(Box::new(ConstOperand {
            span: self.source_info.span,
            user_ty: None,
            const_: Const::Val(
                ConstValue::Scalar(Scalar::from_uint(
                    niche.offset.bytes() as u128,
                    Size::from_bits(128),
                )),
                tcx.types.u128,
            ),
        }));

        let u128_in_niche = self
            .local_decls
            .push(LocalDecl::with_source_info(tcx.types.u128, self.source_info))
            .into();
        let rvalue =
            Rvalue::Cast(CastKind::IntToInt, Operand::Copy(self.value_in_niche), tcx.types.u128);
        self.block_data.statements.push(Statement {
            source_info: self.source_info,
            kind: StatementKind::Assign(Box::new((u128_in_niche, rvalue))),
        });

        self.block_data.terminator = Some(Terminator {
            source_info: self.source_info,
            kind: TerminatorKind::Assert {
                cond: Operand::Copy(self.is_in_range),
                expected: true,
                target: self.new_block,
                msg: Box::new(AssertKind::OccupiedNiche {
                    found: Operand::Copy(u128_in_niche),
                    start,
                    end,
                    type_name,
                    offset,
                    niche_ty: format!("{:?}", niche.value),
                }),
                // This calls panic_misaligned_pointer_dereference, which is #[rustc_nounwind].
                // We never want to insert an unwind into unsafe code, because unwinding could
                // make a failing UB check turn into much worse UB when we start unwinding.
                unwind: UnwindAction::Unreachable,
            },
        });
    }

    fn niche_const(&self, val: u128) -> Operand<'tcx> {
        Operand::Constant(Box::new(ConstOperand {
            span: self.source_info.span,
            user_ty: None,
            const_: Const::Val(
                ConstValue::Scalar(Scalar::from_uint(val, self.niche.size(self.tcx))),
                self.niche.ty(self.tcx),
            ),
        }))
    }

    fn add_assignment(&mut self, place: Place<'tcx>, rvalue: Rvalue<'tcx>) {
        self.block_data.statements.push(Statement {
            source_info: self.source_info,
            kind: StatementKind::Assign(Box::new((place, rvalue))),
        });
    }

    fn check_start_only(&mut self) {
        let start = self.niche_const(self.niche.valid_range.start);

        let rvalue =
            Rvalue::BinaryOp(BinOp::Ge, Box::new((Operand::Copy(self.value_in_niche), start)));
        self.add_assignment(self.is_in_range, rvalue);
    }

    fn check_end_only(&mut self) {
        let end = self.niche_const(self.niche.valid_range.end);

        let rvalue =
            Rvalue::BinaryOp(BinOp::Le, Box::new((Operand::Copy(self.value_in_niche), end)));
        self.add_assignment(self.is_in_range, rvalue);
    }

    fn general_case(&mut self) {
        let mut max = self.niche.valid_range.end.wrapping_sub(self.niche.valid_range.start);
        let size = self.niche.size(self.tcx);
        if size.bits() < 128 {
            let mask = (1 << size.bits()) - 1;
            max &= mask;
        }

        let start = self.niche_const(self.niche.valid_range.start);
        let max_adjusted_allowed_value = self.niche_const(max);

        let biased = self
            .local_decls
            .push(LocalDecl::with_source_info(self.niche.ty(self.tcx), self.source_info))
            .into();
        let rvalue =
            Rvalue::BinaryOp(BinOp::Sub, Box::new((Operand::Copy(self.value_in_niche), start)));
        self.add_assignment(biased, rvalue);

        let rvalue = Rvalue::BinaryOp(
            BinOp::Le,
            Box::new((Operand::Copy(biased), max_adjusted_allowed_value)),
        );
        self.add_assignment(self.is_in_range, rvalue);
    }
}

trait NicheExt {
    fn ty<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Ty<'tcx>;
    fn size<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Size;
}

impl NicheExt for Niche {
    fn ty<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        let types = &tcx.types;
        match self.value {
            Primitive::Int(Integer::I8, _) => types.u8,
            Primitive::Int(Integer::I16, _) => types.u16,
            Primitive::Int(Integer::I32, _) => types.u32,
            Primitive::Int(Integer::I64, _) => types.u64,
            Primitive::Int(Integer::I128, _) => types.u128,
            Primitive::Pointer(_) => types.usize,
            Primitive::F32 => types.u32,
            Primitive::F64 => types.u64,
        }
    }

    fn size<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Size {
        let bits = match self.value {
            Primitive::Int(Integer::I8, _) => 8,
            Primitive::Int(Integer::I16, _) => 16,
            Primitive::Int(Integer::I32, _) => 32,
            Primitive::Int(Integer::I64, _) => 64,
            Primitive::Int(Integer::I128, _) => 128,
            Primitive::Pointer(_) => tcx.sess.target.pointer_width as usize,
            Primitive::F32 => 32,
            Primitive::F64 => 64,
        };
        Size::from_bits(bits)
    }
}
