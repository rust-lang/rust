use crate::MirPass;
use rustc_hir::def::DefKind;
use rustc_hir::lang_items::LangItem;
use rustc_hir::Unsafety;
use rustc_index::IndexVec;
use rustc_middle::mir::interpret::{Pointer, Scalar};
use rustc_middle::mir::visit::NonMutatingUseContext;
use rustc_middle::mir::visit::PlaceContext;
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::*;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{ParamEnv, Ty, TyCtxt, TypeAndMut};
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

        let def_id = body.source.def_id();
        if tcx.type_of(def_id).instantiate_identity().is_coroutine() {
            return;
        }

        // This pass will in general insert an enormous amount of checks, so we need some way to
        // trim them down a bit.
        // Our tactic here is to only emit checks if we are compiling an `unsafe fn` or a function
        // that contains an unsafe block. Yes this means that we can fail to check some operations.
        // If you have a better strategy that doesn't impose 2x compile time overhead, please
        // share.
        let is_safe_code =
            tcx.unsafety_check_result(def_id.expect_local()).used_unsafe_blocks.is_empty();
        let is_unsafe_fn = match tcx.def_kind(def_id) {
            DefKind::Closure => false,
            _ => tcx.fn_sig(def_id).skip_binder().unsafety() == Unsafety::Unsafe,
        };
        if is_safe_code && !is_unsafe_fn {
            return;
        }

        with_no_trimmed_paths!(debug!("Inserting niche checks for {:?}", body.source));

        let basic_blocks = &body.basic_blocks;
        let param_env = tcx.param_env_reveal_all_normalized(def_id);

        // This pass inserts new blocks. Each insertion changes the Location for all
        // statements/blocks after. Iterating or visiting the MIR in order would require updating
        // our current location after every insertion. By iterating backwards, we dodge this issue:
        // The only Locations that an insertion changes have already been handled.
        for block in (0..basic_blocks.len()).rev() {
            let block = block.into();
            let basic_blocks = &body.basic_blocks;
            for statement_index in (0..basic_blocks[block].statements.len()).rev() {
                let location = Location { block, statement_index };
                let statement = &body.basic_blocks[block].statements[statement_index];
                let source_info = statement.source_info;

                let mut finder = NicheFinder { tcx, param_env, body, places: Vec::new() };
                finder.visit_statement(statement, location);
                let places = finder.places;

                let mut checker = NicheChecker { tcx, local_decls: &mut body.local_decls };
                for (place, niche) in places {
                    let basic_blocks = body.basic_blocks.as_mut();
                    let (block_data, new_block) = split_block(basic_blocks, location);
                    checker.insert_niche_check(block_data, new_block, place, niche, source_info);
                }
            }
        }
    }
}

struct NicheFinder<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,
    body: &'a Body<'tcx>,
    places: Vec<(Place<'tcx>, NicheKind)>,
}

impl<'a, 'tcx> Visitor<'tcx> for NicheFinder<'a, 'tcx> {
    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        if let Rvalue::Cast(CastKind::Transmute, op, ty) = rvalue {
            if let Some(niche) = self.get_niche(*ty) && let Some(place) = op.place() {
                with_no_trimmed_paths!(debug!(
                    "Found place {place:?}: {ty:?} with niche {niche:?} due to Transmute: {:?}",
                    self.body.stmt_at(location)
                ));
                self.places.push((place, niche));
            }
        } else {
            self.super_rvalue(rvalue, location);
        }
    }

    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
        match context {
            PlaceContext::NonMutatingUse(
                NonMutatingUseContext::Copy | NonMutatingUseContext::Move,
            ) => {}
            _ => {
                return;
            }
        }

        let ty = place.ty(self.body, self.tcx).ty;
        // bool is actually an i1 to LLVM which means a bool local can never be invalid.
        if ty == self.tcx.types.bool && place.projection.is_empty() {
            return;
        }
        let Some(niche) = self.get_niche(ty) else {
            return;
        };

        with_no_trimmed_paths!(debug!(
            "Found place {place:?}: {ty:?} with niche {niche:?} due to {:?}",
            self.body.stmt_at(location)
        ));
        self.places.push((*place, niche));
    }
}

impl<'a, 'tcx> NicheFinder<'a, 'tcx> {
    fn get_niche(&self, ty: Ty<'tcx>) -> Option<NicheKind> {
        // If we can't get the layout of this, just skip the check.
        let Ok(layout) = self.tcx.layout_of(self.param_env.and(ty)) else {
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

struct NicheChecker<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    local_decls: &'a mut IndexVec<Local, LocalDecl<'tcx>>,
}

impl<'a, 'tcx> NicheChecker<'a, 'tcx> {
    fn insert_niche_check(
        &mut self,
        block_data: &mut BasicBlockData<'tcx>,
        new_block: BasicBlock,
        place: Place<'tcx>,
        niche: NicheKind,
        source_info: SourceInfo,
    ) {
        let mut value_in_niche = self
            .local_decls
            .push(LocalDecl::with_source_info(niche.niche().ty(self.tcx), source_info))
            .into();

        match niche {
            NicheKind::Full(niche) => {
                // The niche occupies the entire source Operand, so we can just transmute
                // directly to the niche primitive.
                let rvalue =
                    Rvalue::Cast(CastKind::Transmute, Operand::Copy(place), niche.ty(self.tcx));
                block_data.statements.push(Statement {
                    source_info,
                    kind: StatementKind::Assign(Box::new((value_in_niche, rvalue))),
                });
            }
            NicheKind::Partial(niche, size) => {
                if niche.offset == Size::ZERO {
                    // FIXME: Delete value_in_niche, we don't need it on this path
                    self.local_decls.pop();

                    // Take the address of the place
                    let full_ptr = self
                        .local_decls
                        .push(LocalDecl::with_source_info(
                            Ty::new_ptr(
                                self.tcx,
                                TypeAndMut {
                                    ty: place.ty(self.local_decls, self.tcx).ty,
                                    mutbl: Mutability::Not,
                                },
                            ),
                            source_info,
                        ))
                        .into();
                    let rvalue = Rvalue::AddressOf(Mutability::Not, place);
                    block_data.statements.push(Statement {
                        source_info,
                        kind: StatementKind::Assign(Box::new((full_ptr, rvalue))),
                    });

                    // Cast to a pointer to the niche type
                    let niche_ptr_ty = Ty::new_ptr(
                        self.tcx,
                        TypeAndMut { ty: niche.ty(self.tcx), mutbl: Mutability::Not },
                    );
                    let niche_ptr = self
                        .local_decls
                        .push(LocalDecl::with_source_info(niche_ptr_ty, source_info))
                        .into();
                    let rvalue =
                        Rvalue::Cast(CastKind::PtrToPtr, Operand::Copy(full_ptr), niche_ptr_ty);
                    block_data.statements.push(Statement {
                        source_info,
                        kind: StatementKind::Assign(Box::new((niche_ptr, rvalue))),
                    });

                    // Set value_in_niche to a projection of our pointer
                    value_in_niche = niche_ptr.project_deeper(&[ProjectionElem::Deref], self.tcx);
                } else {
                    let mu = Ty::new_maybe_uninit(self.tcx, niche.ty(self.tcx));

                    // Transmute the niche-containing type to a [MaybeUninit; N]
                    let array_len = size.bytes() / niche.size(self.tcx).bytes();
                    let mu_array_ty = Ty::new_array(self.tcx, mu, array_len);
                    let mu_array = self
                        .local_decls
                        .push(LocalDecl::with_source_info(mu_array_ty, source_info))
                        .into();
                    let rvalue =
                        Rvalue::Cast(CastKind::Transmute, Operand::Copy(place), mu_array_ty);
                    block_data.statements.push(Statement {
                        source_info,
                        kind: StatementKind::Assign(Box::new((mu_array, rvalue))),
                    });

                    // Convert the niche byte offset into an array index
                    assert_eq!(niche.offset.bytes() % niche.size(self.tcx).bytes(), 0);
                    let offset = niche.offset.bytes() / niche.size(self.tcx).bytes();

                    let niche_as_mu = mu_array.project_deeper(
                        &[ProjectionElem::ConstantIndex {
                            offset,
                            min_length: array_len,
                            from_end: false,
                        }],
                        self.tcx,
                    );

                    // Transmute the MaybeUninit<T> to the niche primitive
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
        .insert_niche_check();
    }
}

#[derive(Clone, Copy, Debug)]
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
    fn insert_niche_check(&mut self) {
        let niche = self.niche;

        let size = niche.size(self.tcx);

        if niche.valid_range.start == 0 {
            // The niche starts at 0, so we can just check if it is Le the end
            self.check_end_only();
        } else if niche.valid_range.end == (u128::MAX >> (128 - size.bits())) {
            // The niche ends at the max, so we can just check if it is Ge the start
            self.check_start_only();
        } else {
            self.general_case();
        }

        let start = self.niche_const(niche.valid_range.start);
        let end = self.niche_const(niche.valid_range.end);

        self.block_data.terminator = Some(Terminator {
            source_info: self.source_info,
            kind: TerminatorKind::Assert {
                cond: Operand::Copy(self.is_in_range),
                expected: true,
                target: self.new_block,
                msg: Box::new(AssertKind::OccupiedNiche {
                    found: Operand::Copy(self.value_in_niche),
                    start,
                    end,
                }),
                // This calls panic_occupied_niche, which is #[rustc_nounwind].
                // We never want to insert an unwind into unsafe code, because unwinding could
                // make a failing UB check turn into much worse UB when we start unwinding.
                unwind: UnwindAction::Unreachable,
            },
        });
    }

    fn niche_const(&self, val: u128) -> Operand<'tcx> {
        let niche_ty = self.niche.ty(self.tcx);
        if niche_ty.is_any_ptr() {
            Operand::Constant(Box::new(ConstOperand {
                span: self.source_info.span,
                user_ty: None,
                const_: Const::Val(
                    ConstValue::Scalar(Scalar::from_maybe_pointer(
                        Pointer::from_addr_invalid(val as u64),
                        &self.tcx,
                    )),
                    niche_ty,
                ),
            }))
        } else {
            Operand::Constant(Box::new(ConstOperand {
                span: self.source_info.span,
                user_ty: None,
                const_: Const::Val(
                    ConstValue::Scalar(Scalar::from_uint(val, self.niche.size(self.tcx))),
                    niche_ty,
                ),
            }))
        }
    }

    fn add_assignment(&mut self, place: Place<'tcx>, rvalue: Rvalue<'tcx>) {
        self.block_data.statements.push(Statement {
            source_info: self.source_info,
            kind: StatementKind::Assign(Box::new((place, rvalue))),
        });
    }

    fn check_start_only(&mut self) {
        let rvalue = if self.niche.valid_range.start == 1 {
            let bound = self.niche_const(0);
            Rvalue::BinaryOp(BinOp::Ne, Box::new((Operand::Copy(self.value_in_niche), bound)))
        } else {
            let bound = self.niche_const(self.niche.valid_range.start);
            Rvalue::BinaryOp(BinOp::Ge, Box::new((Operand::Copy(self.value_in_niche), bound)))
        };
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
            Primitive::Pointer(_) => {
                Ty::new_ptr(tcx, TypeAndMut { ty: types.unit, mutbl: Mutability::Not })
            }
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
