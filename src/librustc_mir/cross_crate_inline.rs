use rustc::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc::mir::{Body, TerminatorKind};
use rustc::ty::query::Providers;
use rustc::ty::TyCtxt;
use rustc_attr::InlineAttr;
use rustc_hir::def_id::DefId;
//use rustc_hir::ItemKind;
//use rustc_session::config::OptLevel;
use rustc_session::config::Sanitizer;

pub fn provide(providers: &mut Providers<'_>) {
    providers.cross_crate_inlinable = cross_crate_inlinable;
}

fn cross_crate_inlinable(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    /*use rustc_hir::Node;
    tcx.hir().as_local_hir_id(def_id).map(|id| match tcx.hir().get(id) {
        Node::Item(item) => match item.kind {
            ItemKind::Static(..) | ItemKind::Const(..) => {
                panic!("STATIC!! {:?}", def_id);
            }
            _ => (),
        },
        _ => (),
    });*/

    let attrs = tcx.codegen_fn_attrs(def_id);

    match attrs.inline {
        InlineAttr::Hint | InlineAttr::Always => return true,
        InlineAttr::Never => return false,
        InlineAttr::None => (),
    }

    if attrs.contains_extern_indicator() {
        return false;
    }

    if tcx.lang_items().start_fn() == Some(def_id) {
        return false;
    }

    // FIXME: Share some of this logic with the MIR inlining pass

    // FIXME: Is this needed?
    if attrs.flags.contains(CodegenFnAttrFlags::TRACK_CALLER) {
        return false;
    }

    // Avoid inlining functions marked as no_sanitize if sanitizer is enabled,
    // since instrumentation might be enabled and performed on the caller.
    match tcx.sess.opts.debugging_opts.sanitizer {
        Some(Sanitizer::Address) => {
            if attrs.flags.contains(CodegenFnAttrFlags::NO_SANITIZE_ADDRESS) {
                return false;
            }
        }
        Some(Sanitizer::Memory) => {
            if attrs.flags.contains(CodegenFnAttrFlags::NO_SANITIZE_MEMORY) {
                return false;
            }
        }
        Some(Sanitizer::Thread) => {
            if attrs.flags.contains(CodegenFnAttrFlags::NO_SANITIZE_THREAD) {
                return false;
            }
        }
        Some(Sanitizer::Leak) => {}
        None => {}
    }

    if tcx.sess.opts.debugging_opts.cross_crate_inline_threshold.is_none() {
        return false;
    }
    /*
        if tcx.sess.opts.optimize != OptLevel::Aggressive {
            return false;
        }
    */
    let r = allow_cross_inline(tcx, def_id, &tcx.optimized_mir(def_id).unwrap_read_only());
    /*if r {
        eprintln!("cross_crate_inlinable({:?})", def_id);
    }*/
    r
}

const THRESHOLD: usize = 30;

const INSTR_COST: usize = 2;
const CALL_PENALTY: usize = 10;
const LANDINGPAD_PENALTY: usize = 5;
const RESUME_PENALTY: usize = 5;

const UNKNOWN_SIZE_COST: usize = 2;

fn allow_cross_inline<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId, body: &Body<'tcx>) -> bool {
    // Generators should have been transformed here.
    debug_assert!(body.yield_ty.is_none());

    let mut threshold =
        tcx.sess.opts.debugging_opts.cross_crate_inline_threshold.unwrap_or(THRESHOLD);

    // Give a bonus functions with a small number of blocks,
    // We normally have two or three blocks for even
    // very small functions.
    if body.basic_blocks().len() <= 3 {
        threshold += threshold / 4;
    }

    let mut cost = 0;

    for block in body.basic_blocks() {
        cost += block.statements.len();

        match block.terminator().kind {
            TerminatorKind::Drop { unwind, .. } | TerminatorKind::DropAndReplace { unwind, .. } => {
                cost += CALL_PENALTY;
                if unwind.is_some() {
                    cost += LANDINGPAD_PENALTY;
                }
            }
            TerminatorKind::Call { cleanup, .. } => {
                cost += CALL_PENALTY;
                if cleanup.is_some() {
                    cost += LANDINGPAD_PENALTY;
                }
            }
            TerminatorKind::Assert { cleanup, .. } => {
                cost += CALL_PENALTY;

                if cleanup.is_some() {
                    cost += LANDINGPAD_PENALTY;
                }
            }
            TerminatorKind::Resume => cost += RESUME_PENALTY,
            _ => cost += INSTR_COST,
        }
    }

    // Count up the cost of local variables and temps, if we know the size
    // use that, otherwise we use a moderately-large dummy cost.

    let ptr_size = tcx.data_layout.pointer_size.bytes();

    let param_env = tcx.param_env(def_id);

    for v in body.vars_and_temps_iter() {
        let v = &body.local_decls[v];
        // Cost of the var is the size in machine-words, if we know
        // it.
        if let Some(size) =
            tcx.layout_of(param_env.and(v.ty)).ok().map(|layout| layout.size.bytes())
        {
            cost += (size / ptr_size) as usize;
        } else {
            cost += UNKNOWN_SIZE_COST;
        }
    }

    cost <= threshold
}
