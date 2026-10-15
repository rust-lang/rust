//! Inserts the checkpoints `RadWriteSets` planned. Runs after `StateTransform`, which has turned
//! each suspension into a write of the coroutine's state followed by a return.

use rustc_abi::VariantIdx;
use rustc_middle::mir::{BasicBlock, Body, CheckpointObject, Depth, StatementKind, TerminatorKind};
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{CoroutineArgs, CoroutineArgsExt, TyCtxt};

use crate::PassPolicy;

pub(super) struct RadAddCheckpoints;

impl<'tcx> crate::MirPass<'tcx> for RadAddCheckpoints {
    fn policy(&self, ctx: &crate::PassCtx<'_>) -> PassPolicy {
        PassPolicy::optional(ctx.opts.unstable_opts.rad_write_sets)
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        // Taking the table keeps it out of optimized MIR and crate metadata.
        let Some(table) = body.coroutine.as_mut().and_then(|info| info.rad_checkpoints.take())
        else {
            return;
        };
        let sites = suspend_sites(body);

        // Make sure that the sites match the actual number of checkpoints
        assert_eq!(sites.len(), table.awaits.len(), "suspensions don't match the plan");

        // TODO: remove debug output when actual pass gets written
        let path = with_no_trimmed_paths!(tcx.def_path_str(body.source.def_id()));
        eprintln!("=== rad checkpoints: {path} ===");

        for (index, (site, checkpoints)) in sites.iter().zip(&table.awaits).enumerate() {
            assert_eq!(site.state.as_usize(), CoroutineArgs::RESERVED_VARIANTS + index);
            emit_checkpoint(tcx, body, site, "checkpoint", &checkpoints.before);
            emit_checkpoint(tcx, body, site, "await effects", &checkpoints.effects);
        }
    }
}

/// Where the coroutine suspends: `StateTransform`'s `discriminant(self) = state; return`
struct SuspendSite {
    block: BasicBlock,
    state: VariantIdx,
}

/// Enumerate the suspend sites by .await()s being called
fn suspend_sites(body: &Body<'_>) -> Vec<SuspendSite> {
    let mut sites: Vec<SuspendSite> = body
        .basic_blocks
        .iter_enumerated()
        .filter_map(|(block, data)| {
            if !matches!(data.terminator().kind, TerminatorKind::Return) {
                return None;
            }
            let StatementKind::SetDiscriminant { variant_index, .. } = data.statements.last()?.kind
            else {
                return None;
            };
            // The reserved states mark returning and panicking, not suspending.
            (variant_index.as_usize() >= CoroutineArgs::RESERVED_VARIANTS)
                .then_some(SuspendSite { block, state: variant_index })
        })
        .collect();
    sites.sort_by_key(|site| site.state);
    sites
}

/// Inserts the `which` checkpoint of the `.await` at `site`, copying `objects`
fn emit_checkpoint<'tcx>(
    _tcx: TyCtxt<'tcx>,
    _body: &mut Body<'tcx>,
    site: &SuspendSite,
    which: &str,
    objects: &[(CheckpointObject, Depth)],
) {
    let objects: Vec<String> = objects
        .iter()
        .map(|(object, depth)| {
            let object = match object {
                CheckpointObject::Discriminant => "state".to_string(),
                CheckpointObject::Saved(slot) => format!("{slot:?}"),
                CheckpointObject::Upvar(Some(field)) => format!("upvar {field:?}"),
                CheckpointObject::Upvar(None) => "upvars".to_string(),
                CheckpointObject::Outside => "outside".to_string(),
            };
            match depth {
                Depth::Shallow => object,
                Depth::Deep => format!("{object} deep"),
            }
        })
        .collect();
    eprintln!(
        "  state {} ({:?}) {which}: copy {}",
        site.state.as_usize(),
        site.block,
        objects.join(", ")
    );
}
