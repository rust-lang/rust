use crate::MirPass;
use rustc_hir::lang_items::LangItem;
use rustc_index::IndexVec;
use rustc_middle::mir::*;
use rustc_middle::mir::{
    interpret::Scalar,
    visit::{MutatingUseContext, NonMutatingUseContext, PlaceContext, Visitor},
};
use rustc_middle::ty::{self, ParamEnv, Ty, TyCtxt, TypeAndMut};
use rustc_session::Session;

pub struct CheckAlignment;

impl<'tcx> MirPass<'tcx> for CheckAlignment {
    fn is_enabled(&self, sess: &Session) -> bool {
        // FIXME(#112480) MSVC and rustc disagree on minimum stack alignment on x86 Windows
        if sess.target.llvm_target == "i686-pc-windows-msvc" {
            return false;
        }
        sess.opts.debug_assertions
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        // This pass emits new panics. If for whatever reason we do not have a panic
        // implementation, running this pass may cause otherwise-valid code to not compile.
        if tcx.lang_items().get(LangItem::PanicImpl).is_none() {
            return;
        }

        let basic_blocks = body.basic_blocks.as_mut();
        let local_decls = &mut body.local_decls;
        let param_env = tcx.param_env_reveal_all_normalized(body.source.def_id());

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

                let mut finder =
                    PointerFinder { tcx, local_decls, param_env, pointers: Vec::new() };
                finder.visit_statement(statement, location);

                for (local, ty) in finder.pointers {
                    debug!("Inserting alignment check for {:?}", ty);
                    let new_block = split_block(basic_blocks, location);
                    insert_alignment_check(
                        tcx,
                        local_decls,
                        &mut basic_blocks[block],
                        local,
                        ty,
                        source_info,
                        new_block,
                    );
                }
            }
        }
    }
}

struct PointerFinder<'tcx, 'a> {
    tcx: TyCtxt<'tcx>,
    local_decls: &'a mut LocalDecls<'tcx>,
    param_env: ParamEnv<'tcx>,
    pointers: Vec<(Place<'tcx>, Ty<'tcx>)>,
}

impl<'tcx, 'a> Visitor<'tcx> for PointerFinder<'tcx, 'a> {
    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
        // We want to only check reads and writes to Places, so we specifically exclude
        // Borrows and AddressOf.
        match context {
            PlaceContext::MutatingUse(
                MutatingUseContext::Store
                | MutatingUseContext::AsmOutput
                | MutatingUseContext::Call
                | MutatingUseContext::Yield
                | MutatingUseContext::Drop,
            ) => {}
            PlaceContext::NonMutatingUse(
                NonMutatingUseContext::Copy | NonMutatingUseContext::Move,
            ) => {}
            _ => {
                return;
            }
        }

        if !place.is_indirect() {
            return;
        }

        // Since Deref projections must come first and only once, the pointer for an indirect place
        // is the Local that the Place is based on.
        let pointer = Place::from(place.local);
        let pointer_ty = self.local_decls[place.local].ty;

        // We only want to check places based on unsafe pointers
        if !pointer_ty.is_unsafe_ptr() {
            trace!("Indirect, but not based on an unsafe ptr, not checking {:?}", place);
            return;
        }

        let pointee_ty =
            pointer_ty.builtin_deref(true).expect("no builtin_deref for an unsafe pointer").ty;
        // Ideally we'd support this in the future, but for now we are limited to sized types.
        if !pointee_ty.is_sized(self.tcx, self.param_env) {
            debug!("Unsafe pointer, but pointee is not known to be sized: {:?}", pointer_ty);
            return;
        }

        // Try to detect types we are sure have an alignment of 1 and skip the check
        // We don't need to look for str and slices, we already rejected unsized types above
        let element_ty = match pointee_ty.kind() {
            ty::Array(ty, _) => *ty,
            _ => pointee_ty,
        };
        if [self.tcx.types.bool, self.tcx.types.i8, self.tcx.types.u8].contains(&element_ty) {
            debug!("Trivially aligned place type: {:?}", pointee_ty);
            return;
        }

        // Ensure that this place is based on an aligned pointer.
        self.pointers.push((pointer, pointee_ty));

        self.super_place(place, context, location);
    }
}

fn split_block(
    basic_blocks: &mut IndexVec<BasicBlock, BasicBlockData<'_>>,
    location: Location,
) -> BasicBlock {
    let block_data = &mut basic_blocks[location.block];

    // Drain every statement after this one and move the current terminator to a new basic block
    let new_block = BasicBlockData {
        statements: block_data.statements.split_off(location.statement_index),
        terminator: block_data.terminator.take(),
        is_cleanup: block_data.is_cleanup,
    };

    basic_blocks.push(new_block)
}

fn insert_alignment_check<'tcx>(
    tcx: TyCtxt<'tcx>,
    local_decls: &mut IndexVec<Local, LocalDecl<'tcx>>,
    block_data: &mut BasicBlockData<'tcx>,
    pointer: Place<'tcx>,
    pointee_ty: Ty<'tcx>,
    source_info: SourceInfo,
    new_block: BasicBlock,
) {
    // Cast the pointer to a *const ()
    let const_raw_ptr = Ty::new_ptr(tcx, TypeAndMut { ty: tcx.types.unit, mutbl: Mutability::Not });
    let rvalue = Rvalue::Cast(CastKind::PtrToPtr, Operand::Copy(pointer), const_raw_ptr);
    let thin_ptr = local_decls.push(LocalDecl::with_source_info(const_raw_ptr, source_info)).into();
    block_data
        .statements
        .push(Statement { source_info, kind: StatementKind::Assign(Box::new((thin_ptr, rvalue))) });

    // Transmute the pointer to a usize (equivalent to `ptr.addr()`)
    let rvalue = Rvalue::Cast(CastKind::Transmute, Operand::Copy(thin_ptr), tcx.types.usize);
    let addr = local_decls.push(LocalDecl::with_source_info(tcx.types.usize, source_info)).into();
    block_data
        .statements
        .push(Statement { source_info, kind: StatementKind::Assign(Box::new((addr, rvalue))) });

    // Get the alignment of the pointee
    let alignment =
        local_decls.push(LocalDecl::with_source_info(tcx.types.usize, source_info)).into();
    let rvalue = Rvalue::NullaryOp(NullOp::AlignOf, pointee_ty);
    block_data.statements.push(Statement {
        source_info,
        kind: StatementKind::Assign(Box::new((alignment, rvalue))),
    });

    // Subtract 1 from the alignment to get the alignment mask
    let alignment_mask =
        local_decls.push(LocalDecl::with_source_info(tcx.types.usize, source_info)).into();
    let one = Operand::Constant(Box::new(ConstOperand {
        span: source_info.span,
        user_ty: None,
        const_: Const::Val(ConstValue::Scalar(Scalar::from_target_usize(1, &tcx)), tcx.types.usize),
    }));
    block_data.statements.push(Statement {
        source_info,
        kind: StatementKind::Assign(Box::new((
            alignment_mask,
            Rvalue::BinaryOp(BinOp::Sub, Box::new((Operand::Copy(alignment), one))),
        ))),
    });

    // BitAnd the alignment mask with the pointer
    let alignment_bits =
        local_decls.push(LocalDecl::with_source_info(tcx.types.usize, source_info)).into();
    block_data.statements.push(Statement {
        source_info,
        kind: StatementKind::Assign(Box::new((
            alignment_bits,
            Rvalue::BinaryOp(
                BinOp::BitAnd,
                Box::new((Operand::Copy(addr), Operand::Copy(alignment_mask))),
            ),
        ))),
    });

    // Check if the alignment bits are all zero
    let is_ok = local_decls.push(LocalDecl::with_source_info(tcx.types.bool, source_info)).into();
    let zero = Operand::Constant(Box::new(ConstOperand {
        span: source_info.span,
        user_ty: None,
        const_: Const::Val(ConstValue::Scalar(Scalar::from_target_usize(0, &tcx)), tcx.types.usize),
    }));
    block_data.statements.push(Statement {
        source_info,
        kind: StatementKind::Assign(Box::new((
            is_ok,
            Rvalue::BinaryOp(BinOp::Eq, Box::new((Operand::Copy(alignment_bits), zero.clone()))),
        ))),
    });

    // Set this block's terminator to our assert, continuing to new_block if we pass
    block_data.terminator = Some(Terminator {
        source_info,
        kind: TerminatorKind::Assert {
            cond: Operand::Copy(is_ok),
            expected: true,
            target: new_block,
            msg: Box::new(AssertKind::MisalignedPointerDereference {
                required: Operand::Copy(alignment),
                found: Operand::Copy(addr),
            }),
            // This calls panic_misaligned_pointer_dereference, which is #[rustc_nounwind].
            // We never want to insert an unwind into unsafe code, because unwinding could
            // make a failing UB check turn into much worse UB when we start unwinding.
            unwind: UnwindAction::Unreachable,
        },
    });
}
