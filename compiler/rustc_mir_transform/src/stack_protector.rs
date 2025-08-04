//! Validates the MIR to ensure that invariants are upheld.

use std::ops::Deref;

use rustc_middle::mir::*;
use rustc_middle::ty;
use rustc_middle::ty::{Instance, TyCtxt};
use rustc_target::callconv::PassMode;

pub(super) struct StackProtectorFinder;

impl<'tcx> crate::MirPass<'tcx> for StackProtectorFinder {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        use Rvalue::*;
        let def_id = body.source.def_id();

        for block in body.basic_blocks.iter() {
            for stmt in block.statements.iter() {
                if let StatementKind::Assign(assign) = &stmt.kind {
                    let (_, rvalue) = assign.deref();
                    match rvalue {
                        // Get a reference/pointer to a variable
                        Ref(..) | ThreadLocalRef(_) | RawPtr(..) => {
                            tcx.stack_protector.borrow_mut().insert(def_id);
                            return;
                        }
                        _ => continue,
                    }
                }
            }

            if let Some(terminator) = block.terminator.as_ref() {
                if let TerminatorKind::Call { destination: place, .. } = &terminator.kind {
                    // Returns a mutable raw pointer, possibly a memory allocation function
                    if let ty::RawPtr(_, Mutability::Mut) = place.ty(body, tcx).ty.kind() {
                        tcx.stack_protector.borrow_mut().insert(def_id);
                        return;
                    }
                }
            }

            let instance = Instance::mono(tcx, def_id);
            let Ok(fn_abi) = tcx.fn_abi_of_instance(
                ty::TypingEnv::fully_monomorphized().as_query_input((instance, ty::List::empty())),
            ) else {
                // FIXME: Find when an Err() message is returned

                return;
            };

            // for arg in fn_abi.args.iter() {
            //     if matches!(&arg.mode, PassMode::Indirect { attrs: _, meta_attrs: _, on_stack: false }) {
            //         tcx.stack_protector.borrow_mut().insert(def_id);
            //         return;
            //     }
            // }

            let ret = &fn_abi.ret;
            if matches!(&ret.mode, PassMode::Indirect { attrs: _, meta_attrs: _, on_stack: false })
            {
                tcx.stack_protector.borrow_mut().insert(def_id);
                return;
            }
        }
    }

    fn is_required(&self) -> bool {
        true
    }
}
