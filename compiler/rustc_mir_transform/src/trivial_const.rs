use std::ops::Deref;

use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalDefId;
use rustc_middle::mir::{
    Body, Const, ConstValue, Operand, Place, RETURN_PLACE, Rvalue, START_BLOCK, StatementKind,
    TerminatorKind, UnevaluatedConst,
};
use rustc_middle::ty::{Ty, TyCtxt, TypeVisitableExt};

/// If the given def is a trivial const, returns the value and type the const evaluates to.
///
/// A "trivial const" is a const which can be easily proven to evaluate successfully, and the value
/// that it evaluates to can be easily found without going through the usual MIR phases for a const.
///
/// Currently, we support two forms of trivial const.
///
/// The base case is this:
/// ```
/// const A: usize = 0;
/// ```
/// which has this MIR:
/// ```text
/// const A: usize = {
///     let mut _0: usize;
///
///     bb0: {
///         _0 = const 0_usize;
///         return;
///     }
/// }
/// ```
/// Which we recognize by looking for a Body which has a single basic block with a return
/// terminator and a single statement which assigns an `Operand::Constant(Const::Val)` to the
/// return place.
/// This scenario meets the required criteria because:
/// * Control flow cannot panic, we don't have any calls or assert terminators
/// * The value of the const is already computed, so it cannot fail
///
/// In addition to assignment of literals, assignments of trivial consts are also considered
/// trivial consts. In this case, both `A` and `B` are trivial:
/// ```
/// const A: usize = 0;
/// const B: usize = A;
/// ```
pub(crate) fn trivial_const<'a, 'tcx: 'a, F, B>(
    tcx: TyCtxt<'tcx>,
    def: LocalDefId,
    body_provider: F,
) -> Option<(ConstValue, Ty<'tcx>)>
where
    F: FnOnce() -> B,
    B: Deref<Target = Body<'tcx>>,
{
    if !matches!(tcx.def_kind(def), DefKind::AssocConst | DefKind::Const | DefKind::AnonConst) {
        return None;
    }

    let body = body_provider();

    if body.has_opaque_types() {
        return None;
    }

    if body.basic_blocks.len() != 1 {
        return None;
    }

    let block = &body.basic_blocks[START_BLOCK];
    if block.statements.len() != 1 {
        return None;
    }

    if block.terminator().kind != TerminatorKind::Return {
        return None;
    }

    let StatementKind::Assign(box (place, rvalue)) = &block.statements[0].kind else {
        return None;
    };

    if *place != Place::from(RETURN_PLACE) {
        return None;
    }

    let Rvalue::Use(Operand::Constant(c)) = rvalue else {
        return None;
    };
    match c.const_ {
        Const::Ty(..) => None,
        Const::Unevaluated(UnevaluatedConst { def, args, .. }, _ty) => {
            if !args.is_empty() {
                return None;
            }
            tcx.trivial_const(def)
        }
        Const::Val(v, ty) => Some((v, ty)),
    }
}

// The query provider is based on calling the free function trivial_const, which calls mir_built,
// which internally has a fast-path for trivial consts so it too calls trivial_const. This isn't
// recursive, but we are checking if the const is trivial twice. A better design might detect
// trivial consts before getting to MIR, which would hopefully straighten this out.
pub(crate) fn trivial_const_provider<'tcx>(
    tcx: TyCtxt<'tcx>,
    def: LocalDefId,
) -> Option<(ConstValue, Ty<'tcx>)> {
    trivial_const(tcx, def, || tcx.mir_built(def).borrow())
}
