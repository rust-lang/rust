use std::borrow::Cow;

use rustc_middle::mir::{Body, MirPhase};
use rustc_middle::ty::TyCtxt;

use super::{dump_mir, validate};

/// A streamlined trait that you can implement to create a pass; the
/// pass will be named after the type, and it will consist of a main
/// loop that goes over each available MIR and applies `run_pass`.
pub trait MirPass<'tcx> {
    const LEVEL: OptLevel;

    fn name(&self) -> Cow<'_, str> {
        let name = std::any::type_name::<Self>();
        name.rsplit(':').next().unwrap().into()
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>);
}

const UNSOUND_MIR_OPT_LEVEL: u8 = 1;

pub enum OptLevel {
    /// Passes that will run at `-Zmir-opt-level=N` or higher.
    N(u8),

    /// Passes that are known or suspected to cause miscompilations.
    ///
    /// These passes only run if `-Zunsound-mir-opts` is enabled and we are at `mir-opt-level=1` or
    /// above.
    Unsound,

    /// Passes that clean up the MIR after other transformations.
    ///
    /// A `Cleanup` pass is skipped if the pass immediately preceding it is skipped.
    //
    // FIXME: Maybe we want to run the cleanup unless *all* passes between it and the last cleanup
    // were skipped?
    Cleanup,
}

impl OptLevel {
    /// A pass that will always run, regardless of `-Zmir-opt-level` or other flags.
    pub const ALWAYS: Self = OptLevel::N(0);

    /// A pass that will run if no flags are passed to `rustc` (e.g. `-Zmir-opt-level`).
    pub const DEFAULT: Self = OptLevel::N(1);
}

/// RAII-style type that runs a series of optimization passes on a `Body`.
///
/// Upon going out of scope, this type updates the `MirPhase` of that `Body`.
pub struct PassManager<'mir, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'mir mut Body<'tcx>,
    phase_change: MirPhase,
    passes_run: usize,
    skip_cleanup: bool,
}

impl PassManager<'mir, 'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, body: &'mir mut Body<'tcx>, phase_change: MirPhase) -> Self {
        assert!(body.phase < phase_change);

        let ret = PassManager { tcx, phase_change, body, passes_run: 0, skip_cleanup: false };

        if tcx.sess.opts.debugging_opts.validate_mir {
            ret.validate(&format!("input to phase {:?}", phase_change));
        }

        ret
    }

    fn is_pass_enabled(&self, level: OptLevel) -> bool {
        let opts = &self.tcx.sess.opts.debugging_opts;
        let level_required = match level {
            OptLevel::Cleanup => return !self.skip_cleanup,
            OptLevel::Unsound if !opts.unsound_mir_opts => return false,

            OptLevel::Unsound => UNSOUND_MIR_OPT_LEVEL,
            OptLevel::N(n) => n,
        };

        opts.mir_opt_level >= level_required as usize
    }

    pub fn validate(&self, when: &str) {
        validate::validate_body(self.tcx, self.body, when);
    }

    pub fn run_pass<P>(&mut self, pass: &P)
    where
        P: MirPass<'tcx>,
    {
        if !self.is_pass_enabled(pass) {
            info!("Skipping {}", pass.name());
            self.skip_cleanup = true;
            return;
        }

        dump_mir::on_mir_pass(
            self.tcx,
            &format_args!("{:03}-{:03}", self.body.phase.phase_index(), self.passes_run),
            &pass.name(),
            self.body,
            false,
        );
        pass.run_pass(self.tcx, self.body);
        dump_mir::on_mir_pass(
            self.tcx,
            &format_args!("{:03}-{:03}", self.body.phase.phase_index(), self.passes_run),
            &pass.name(),
            self.body,
            true,
        );

        self.skip_cleanup = false;
        self.passes_run += 1;

        if self.tcx.sess.opts.debugging_opts.validate_mir {
            self.validate(&format!(
                "after {} while transitioning to {:?}",
                pass.name(),
                self.phase_change
            ));
        }
    }
}

impl Drop for PassManager<'mir, 'tcx> {
    fn drop(&mut self) {
        self.body.phase = self.phase_change;

        // Do MIR validation after all optimization passes have run regardless of `-Zvalidate_mir`.
        if self.phase_change == MirPhase::Optimization {
            self.validate(&format!("end of phase {:?}", self.phase_change));
        }
    }
}

macro_rules! run_passes {
    ($manager:expr => [$($pass:expr),* $(,)?]) => {{
        let ref mut manager: PassManager<'_, '_> = $manager;
        $( manager.run_pass(&$pass); )*
    }}
}
