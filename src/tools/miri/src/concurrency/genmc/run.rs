use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;

use genmc_sys::EstimationResult;
use rustc_log::tracing;
use rustc_middle::ty::TyCtxt;

use super::GlobalState;
use crate::concurrency::genmc::ExecutionEndResult;
use crate::rustc_const_eval::interpret::PointerArithmetic;
use crate::{GenmcConfig, GenmcCtx, MiriConfig};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum GenmcMode {
    Estimation,
    Verification,
}

/// Do a complete run of the program in GenMC mode.
/// This will call `eval_entry` multiple times, until either:
/// - An error is detected (indicated by a `None` return value)
/// - All possible executions are explored.
///
/// Returns `None` is an error is detected, or `Some(return_value)` with the return value of the last run of the program.
pub fn run_genmc_mode<'tcx>(
    config: &MiriConfig,
    eval_entry: impl Fn(Rc<GenmcCtx>) -> Option<i32>,
    tcx: TyCtxt<'tcx>,
) -> Option<i32> {
    let genmc_config = config.genmc_config.as_ref().unwrap();
    // Run in Estimation mode if requested.
    if genmc_config.do_estimation {
        eprintln!("Estimating GenMC verification time...");
        run_genmc_mode_impl(config, &eval_entry, tcx, GenmcMode::Estimation)?;
    }
    // Run in Verification mode.
    eprintln!("Running GenMC Verification...");
    run_genmc_mode_impl(config, &eval_entry, tcx, GenmcMode::Verification)
}

fn run_genmc_mode_impl<'tcx>(
    config: &MiriConfig,
    eval_entry: &impl Fn(Rc<GenmcCtx>) -> Option<i32>,
    tcx: TyCtxt<'tcx>,
    mode: GenmcMode,
) -> Option<i32> {
    let time_start = Instant::now();
    let genmc_config = config.genmc_config.as_ref().unwrap();

    // There exists only one `global_state` per full run in GenMC mode.
    // It is shared by all `GenmcCtx` in this run.
    // FIXME(genmc): implement multithreading once GenMC supports it.
    let global_state = Arc::new(GlobalState::new(tcx.target_usize_max()));
    let genmc_ctx = Rc::new(GenmcCtx::new(config, global_state, mode));

    // `rep` is used to report the progress, Miri will panic on wrap-around.
    for rep in 0u64.. {
        tracing::info!("Miri-GenMC loop {}", rep + 1);

        // Prepare for the next execution and inform GenMC about it.
        genmc_ctx.prepare_next_execution();

        // Execute the program until completion to get the return value, or return if an error happens:
        let Some(return_code) = eval_entry(genmc_ctx.clone()) else {
            genmc_ctx.print_genmc_output(genmc_config, tcx);
            return None;
        };

        // We inform GenMC that the execution is complete.
        // If there was an error, we print it.
        match genmc_ctx.handle_execution_end() {
            ExecutionEndResult::Continue => continue,
            ExecutionEndResult::Stop => {
                let elapsed_time_sec = Instant::now().duration_since(time_start).as_secs_f64();
                // Print the output for the current mode.
                if mode == GenmcMode::Estimation {
                    genmc_ctx.print_estimation_output(genmc_config, elapsed_time_sec);
                } else {
                    genmc_ctx.print_verification_output(genmc_config, elapsed_time_sec);
                }
                // Return the return code of the last execution.
                return Some(return_code);
            }
            ExecutionEndResult::Error(error) => {
                // This can be reached for errors that affect the entire execution, not just a specific event.
                // For instance, linearizability checking and liveness checking report their errors this way.
                // Neither are supported by Miri-GenMC at the moment though. However, GenMC also
                // treats races on deallocation as global errors, so this code path is still reachable.
                // Since we don't have any span information for the error at this point,
                // we just print GenMC's error string, and the full GenMC output if requested.
                eprintln!("(GenMC) Error detected: {error}");
                genmc_ctx.print_genmc_output(genmc_config, tcx);
                return None;
            }
        }
    }
    unreachable!()
}

impl GenmcCtx {
    /// Print the full output message produced by GenMC if requested, or a hint on how to enable it.
    ///
    /// This message can be very verbose and is likely not useful for the average user.
    /// This function should be called *after* Miri has printed all of its output.
    fn print_genmc_output(&self, genmc_config: &GenmcConfig, tcx: TyCtxt<'_>) {
        if genmc_config.print_genmc_output {
            eprintln!("GenMC error report:");
            eprintln!("{}", self.get_result_message());
        } else {
            tcx.dcx().note(
                "add `-Zmiri-genmc-print-genmc-output` to MIRIFLAGS to see the detailed GenMC error report"
            );
        }
    }

    /// Given the time taken for the estimation mode run, print the expected time range for verification.
    /// Verbose output also includes information about the expected number of executions and how many estimation rounds were explored or got blocked.
    fn print_estimation_output(&self, genmc_config: &GenmcConfig, elapsed_time_sec: f64) {
        let EstimationResult { mean, sd, explored_execs, blocked_execs } =
            self.get_estimation_results();
        #[allow(clippy::as_conversions)]
        let time_per_exec_sec = elapsed_time_sec / (explored_execs as f64 + blocked_execs as f64);
        let estimated_mean_sec = time_per_exec_sec * mean;
        let estimated_sd_sec = time_per_exec_sec * sd;

        if genmc_config.verbose_output {
            eprintln!("Finished estimation in {elapsed_time_sec:.2?}s");
            if blocked_execs != 0 {
                eprintln!("  Explored executions: {explored_execs}");
                eprintln!("  Blocked  executions: {blocked_execs}");
            }
            eprintln!("Expected number of executions: {mean:.0} ± {sd:.0}");
        }
        // The estimation can be out-of-bounds of an `f64`.
        if !(mean.is_finite() && mean >= 0.0 && sd.is_finite() && sd >= 0.0) {
            eprintln!("WARNING: Estimation gave weird results, there may have been an overflow.");
        }
        eprintln!("Expected verification time: {estimated_mean_sec:.2}s ± {estimated_sd_sec:.2}s");
    }

    /// Given the time taken for the verification mode run, print the expected time range for verification.
    /// Verbose output also includes information about the expected number of executions and how many estimation rounds were explored or got blocked.
    fn print_verification_output(&self, genmc_config: &GenmcConfig, elapsed_time_sec: f64) {
        let explored_execution_count = self.get_explored_execution_count();
        let blocked_execution_count = self.get_blocked_execution_count();
        eprintln!(
            "Verification complete with {} executions. No errors found.",
            explored_execution_count + blocked_execution_count
        );
        if genmc_config.verbose_output {
            if blocked_execution_count > 0 {
                eprintln!("Number of complete executions explored: {explored_execution_count}");
                eprintln!("Number of blocked executions seen: {blocked_execution_count}");
            }
            eprintln!("Verification took {elapsed_time_sec:.2?}s.");
        }
    }
}
