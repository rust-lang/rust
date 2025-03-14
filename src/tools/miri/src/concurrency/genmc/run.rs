use std::rc::Rc;
use std::sync::Arc;

use rustc_middle::ty::TyCtxt;

use super::GlobalState;
use crate::rustc_const_eval::interpret::PointerArithmetic;
use crate::{GenmcCtx, MiriConfig};

/// Do a complete run of the program in GenMC mode.
/// This will call `eval_entry` multiple times, until either:
/// - An error is detected (indicated by a `None` return value)
/// - All possible executions are explored.
///
/// FIXME(genmc): add estimation mode setting.
pub fn run_genmc_mode<'tcx>(
    config: &MiriConfig,
    eval_entry: impl Fn(Rc<GenmcCtx>) -> Option<i32>,
    tcx: TyCtxt<'tcx>,
) -> Option<i32> {
    let genmc_config = config.genmc_config.as_ref().unwrap();

    // There exists only one `global_state` per full run in GenMC mode.
    // It is shared by all `GenmcCtx` in this run.
    // FIXME(genmc): implement multithreading once GenMC supports it.
    let global_state = Arc::new(GlobalState::new(tcx.target_usize_max()));
    let genmc_ctx = Rc::new(GenmcCtx::new(config, global_state));

    // `rep` is used to report the progress, Miri will panic on wrap-around.
    for rep in 0u64.. {
        tracing::info!("Miri-GenMC loop {}", rep + 1);

        // Prepare for the next execution and inform GenMC about it.
        genmc_ctx.prepare_next_execution();

        // Execute the program until completion to get the return value, or return if an error happens:
        let Some(return_code) = eval_entry(genmc_ctx.clone()) else {
            // If requested, print the output GenMC produced:
            if genmc_config.print_genmc_output {
                eprintln!("== raw GenMC output =========================");
                eprintln!("{}", genmc_ctx.get_result_message());
                eprintln!("== end of raw GenMC output ==================");
            }
            return None;
        };

        // We inform GenMC that the execution is complete. If there was an error, we print it.
        if let Some(error) = genmc_ctx.handle_execution_end() {
            // This can be reached for errors that affect the entire execution, not just a specific event.
            // For instance, linearizability checking and liveness checking report their errors this way.
            // Neither are supported by Miri-GenMC at the moment though. However, GenMC also
            // treats races on deallocation as global errors, so this code path is still reachable.
            // Since we don't have any span information for the error at this point,
            // we just print GenMC's error message.
            eprintln!("(GenMC) Error detected: {error}");
            eprintln!();
            eprintln!("{}", genmc_ctx.get_result_message());
            return None;
        }

        // Check if we've explored enough executions:
        if !genmc_ctx.is_exploration_done() {
            continue;
        }

        eprintln!("(GenMC) Verification complete. No errors were detected.");

        let explored_execution_count = genmc_ctx.get_explored_execution_count();
        let blocked_execution_count = genmc_ctx.get_blocked_execution_count();

        eprintln!("Number of complete executions explored: {explored_execution_count}");
        if blocked_execution_count > 0 {
            eprintln!("Number of blocked executions seen: {blocked_execution_count}");
        }

        // Return the return code of the last execution.
        return Some(return_code);
    }
    unreachable!()
}
