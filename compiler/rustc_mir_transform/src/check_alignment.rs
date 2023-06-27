use crate::MirPass;
use rustc_hir::def_id::DefId;
use rustc_hir::lang_items::LangItem;
use rustc_index::IndexVec;
use rustc_middle::mir::*;
use rustc_middle::mir::{
    interpret::{ConstValue, Scalar},
    visit::{PlaceContext, Visitor},
};
use rustc_middle::ty::{Ty, TyCtxt, TypeAndMut};
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

        for block in (0..basic_blocks.len()).rev() {
            let block = block.into();
            for statement_index in (0..basic_blocks[block].statements.len()).rev() {
                let location = Location { block, statement_index };
                let statement = &basic_blocks[block].statements[statement_index];
                let source_info = statement.source_info;

                let mut finder = PointerFinder {
                    local_decls,
                    tcx,
                    pointers: Vec::new(),
                    def_id: body.source.def_id(),
                };
                for (pointer, pointee_ty) in finder.find_pointers(statement) {
                    debug!("Inserting alignment check for {:?}", pointer.ty(&*local_decls, tcx).ty);

                    let new_block = split_block(basic_blocks, location);
                    insert_alignment_check(
                        tcx,
                        local_decls,
                        &mut basic_blocks[block],
                        pointer,
                        pointee_ty,
                        source_info,
                        new_block,
                    );
                }
            }
        }
    }
}

impl<'tcx, 'a> PointerFinder<'tcx, 'a> {
    fn find_pointers(&mut self, statement: &Statement<'tcx>) -> Vec<(Place<'tcx>, Ty<'tcx>)> {
        self.pointers.clear();
        self.visit_statement(statement, Location::START);
        core::mem::take(&mut self.pointers)
    }
}

struct PointerFinder<'tcx, 'a> {
    local_decls: &'a mut LocalDecls<'tcx>,
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    pointers: Vec<(Place<'tcx>, Ty<'tcx>)>,
}

impl<'tcx, 'a> Visitor<'tcx> for PointerFinder<'tcx, 'a> {
    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        if let Rvalue::AddressOf(..) = rvalue {
            // Ignore dereferences inside of an AddressOf
            return;
        }
        self.super_rvalue(rvalue, location);
    }

    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, _location: Location) {
        if let PlaceContext::NonUse(_) = context {
            return;
        }
        if !place.is_indirect() {
            return;
        }

        let pointer = Place::from(place.local);
        let pointer_ty = pointer.ty(&*self.local_decls, self.tcx).ty;

        // We only want to check unsafe pointers
        if !pointer_ty.is_unsafe_ptr() {
            trace!("Indirect, but not an unsafe ptr, not checking {:?}", pointer_ty);
            return;
        }

        let Some(pointee) = pointer_ty.builtin_deref(true) else {
            debug!("Indirect but no builtin deref: {:?}", pointer_ty);
            return;
        };
        let mut pointee_ty = pointee.ty;
        if pointee_ty.is_array() || pointee_ty.is_slice() || pointee_ty.is_str() {
            pointee_ty = pointee_ty.sequence_element_type(self.tcx);
        }

        if !pointee_ty.is_sized(self.tcx, self.tcx.param_env_reveal_all_normalized(self.def_id)) {
            debug!("Unsafe pointer, but unsized: {:?}", pointer_ty);
            return;
        }

        if [self.tcx.types.bool, self.tcx.types.i8, self.tcx.types.u8, self.tcx.types.str_]
            .contains(&pointee_ty)
        {
            debug!("Trivially aligned pointee type: {:?}", pointer_ty);
            return;
        }

        self.pointers.push((pointer, pointee_ty))
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
    let const_raw_ptr = tcx.mk_ptr(TypeAndMut { ty: tcx.types.unit, mutbl: Mutability::Not });
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
    let one = Operand::Constant(Box::new(Constant {
        span: source_info.span,
        user_ty: None,
        literal: ConstantKind::Val(
            ConstValue::Scalar(Scalar::from_target_usize(1, &tcx)),
            tcx.types.usize,
        ),
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
    let zero = Operand::Constant(Box::new(Constant {
        span: source_info.span,
        user_ty: None,
        literal: ConstantKind::Val(
            ConstValue::Scalar(Scalar::from_target_usize(0, &tcx)),
            tcx.types.usize,
        ),
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
