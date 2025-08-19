use rustc_abi::Align;
use rustc_index::IndexVec;
use rustc_middle::mir::interpret::Scalar;
use rustc_middle::mir::visit::PlaceContext;
use rustc_middle::mir::*;
use rustc_middle::ty::{Ty, TyCtxt};
use rustc_session::Session;

use crate::check_pointers::{BorrowedFieldProjectionMode, PointerCheck, check_pointers};

pub(super) struct CheckAlignment;

impl<'tcx> crate::MirPass<'tcx> for CheckAlignment {
    fn is_enabled(&self, sess: &Session) -> bool {
        sess.ub_checks()
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        // Skip trivially aligned place types.
        let excluded_pointees = [tcx.types.bool, tcx.types.i8, tcx.types.u8];

        // When checking the alignment of references to field projections (`&(*ptr).a`),
        // we need to make sure that the reference is aligned according to the field type
        // and not to the pointer type.
        check_pointers(
            tcx,
            body,
            &excluded_pointees,
            insert_alignment_check,
            BorrowedFieldProjectionMode::FollowProjections,
        );
    }

    fn is_required(&self) -> bool {
        true
    }
}

/// Inserts the actual alignment check's logic. Returns a
/// [AssertKind::MisalignedPointerDereference] on failure.
fn insert_alignment_check<'tcx>(
    tcx: TyCtxt<'tcx>,
    pointer: Place<'tcx>,
    pointee_ty: Ty<'tcx>,
    _context: PlaceContext,
    local_decls: &mut IndexVec<Local, LocalDecl<'tcx>>,
    stmts: &mut Vec<Statement<'tcx>>,
    source_info: SourceInfo,
) -> PointerCheck<'tcx> {
    // Cast the pointer to a *const ().
    let const_raw_ptr = Ty::new_imm_ptr(tcx, tcx.types.unit);
    let rvalue = Rvalue::Cast(CastKind::PtrToPtr, Operand::Copy(pointer), const_raw_ptr);
    let thin_ptr = local_decls.push(LocalDecl::with_source_info(const_raw_ptr, source_info)).into();
    stmts.push(Statement::new(source_info, StatementKind::Assign(Box::new((thin_ptr, rvalue)))));

    // Transmute the pointer to a usize (equivalent to `ptr.addr()`).
    let rvalue = Rvalue::Cast(CastKind::Transmute, Operand::Copy(thin_ptr), tcx.types.usize);
    let addr = local_decls.push(LocalDecl::with_source_info(tcx.types.usize, source_info)).into();
    stmts.push(Statement::new(source_info, StatementKind::Assign(Box::new((addr, rvalue)))));

    // Get the alignment of the pointee
    let alignment =
        local_decls.push(LocalDecl::with_source_info(tcx.types.usize, source_info)).into();
    let rvalue = Rvalue::NullaryOp(NullOp::AlignOf, pointee_ty);
    stmts.push(Statement::new(source_info, StatementKind::Assign(Box::new((alignment, rvalue)))));

    // Subtract 1 from the alignment to get the alignment mask
    let alignment_mask =
        local_decls.push(LocalDecl::with_source_info(tcx.types.usize, source_info)).into();
    let one = Operand::Constant(Box::new(ConstOperand {
        span: source_info.span,
        user_ty: None,
        const_: Const::Val(ConstValue::Scalar(Scalar::from_target_usize(1, &tcx)), tcx.types.usize),
    }));
    stmts.push(Statement::new(
        source_info,
        StatementKind::Assign(Box::new((
            alignment_mask,
            Rvalue::BinaryOp(BinOp::Sub, Box::new((Operand::Copy(alignment), one))),
        ))),
    ));

    // If this target does not have reliable alignment, further limit the mask by anding it with
    // the mask for the highest reliable alignment.
    #[allow(irrefutable_let_patterns)]
    if let max_align = tcx.sess.target.max_reliable_alignment()
        && max_align < Align::MAX
    {
        let max_mask = max_align.bytes() - 1;
        let max_mask = Operand::Constant(Box::new(ConstOperand {
            span: source_info.span,
            user_ty: None,
            const_: Const::Val(
                ConstValue::Scalar(Scalar::from_target_usize(max_mask, &tcx)),
                tcx.types.usize,
            ),
        }));
        stmts.push(Statement::new(
            source_info,
            StatementKind::Assign(Box::new((
                alignment_mask,
                Rvalue::BinaryOp(
                    BinOp::BitAnd,
                    Box::new((Operand::Copy(alignment_mask), max_mask)),
                ),
            ))),
        ));
    }

    // BitAnd the alignment mask with the pointer
    let alignment_bits =
        local_decls.push(LocalDecl::with_source_info(tcx.types.usize, source_info)).into();
    stmts.push(Statement::new(
        source_info,
        StatementKind::Assign(Box::new((
            alignment_bits,
            Rvalue::BinaryOp(
                BinOp::BitAnd,
                Box::new((Operand::Copy(addr), Operand::Copy(alignment_mask))),
            ),
        ))),
    ));

    // Check if the alignment bits are all zero
    let is_ok = local_decls.push(LocalDecl::with_source_info(tcx.types.bool, source_info)).into();
    let zero = Operand::Constant(Box::new(ConstOperand {
        span: source_info.span,
        user_ty: None,
        const_: Const::Val(ConstValue::Scalar(Scalar::from_target_usize(0, &tcx)), tcx.types.usize),
    }));
    stmts.push(Statement::new(
        source_info,
        StatementKind::Assign(Box::new((
            is_ok,
            Rvalue::BinaryOp(BinOp::Eq, Box::new((Operand::Copy(alignment_bits), zero.clone()))),
        ))),
    ));

    // Emit a check that asserts on the alignment and otherwise triggers a
    // AssertKind::MisalignedPointerDereference.
    PointerCheck {
        cond: Operand::Copy(is_ok),
        assert_kind: Box::new(AssertKind::MisalignedPointerDereference {
            required: Operand::Copy(alignment),
            found: Operand::Copy(addr),
        }),
    }
}
