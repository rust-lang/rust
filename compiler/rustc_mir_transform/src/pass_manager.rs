use std::borrow::Cow;

use rustc_middle::mir::{self, Body, MirPhase, RuntimePhase};
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;

use crate::{validate, MirPass};

/// Just like `MirPass`, except it cannot mutate `Body`.
pub trait MirLint<'tcx> {
    fn name(&self) -> Cow<'_, str> {
        let name = std::any::type_name::<Self>();
        if let Some(tail) = name.rfind(':') {
            Cow::from(&name[tail + 1..])
        } else {
            Cow::from(name)
        }
    }

    fn is_enabled(&self, _sess: &Session) -> bool {
        true
    }

    fn run_lint(&self, tcx: TyCtxt<'tcx>, body: &Body<'tcx>);
}

/// An adapter for `MirLint`s that implements `MirPass`.
#[derive(Debug, Clone)]
pub struct Lint<T>(pub T);

impl<'tcx, T> MirPass<'tcx> for Lint<T>
where
    T: MirLint<'tcx>,
{
    fn name(&self) -> Cow<'_, str> {
        self.0.name()
    }

    fn is_enabled(&self, sess: &Session) -> bool {
        self.0.is_enabled(sess)
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        self.0.run_lint(tcx, body)
    }

    fn is_mir_dump_enabled(&self) -> bool {
        false
    }
}

pub struct WithMinOptLevel<T>(pub u32, pub T);

impl<'tcx, T> MirPass<'tcx> for WithMinOptLevel<T>
where
    T: MirPass<'tcx>,
{
    fn name(&self) -> Cow<'_, str> {
        self.1.name()
    }

    fn is_enabled(&self, sess: &Session) -> bool {
        sess.mir_opt_level() >= self.0 as usize
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        self.1.run_pass(tcx, body)
    }

    fn phase_change(&self) -> Option<MirPhase> {
        self.1.phase_change()
    }
}

/// Run the sequence of passes without validating the MIR after each pass. The MIR is still
/// validated at the end.
pub fn run_passes_no_validate<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    passes: &[&dyn MirPass<'tcx>],
) {
    run_passes_inner(tcx, body, passes, false);
}

pub fn run_passes<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>, passes: &[&dyn MirPass<'tcx>]) {
    run_passes_inner(tcx, body, passes, true);
}

fn run_passes_inner<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    passes: &[&dyn MirPass<'tcx>],
    validate_each: bool,
) {
    let start_phase = body.phase;
    let mut cnt = 0;

    let validate = validate_each & tcx.sess.opts.unstable_opts.validate_mir;
    let overridden_passes = &tcx.sess.opts.unstable_opts.mir_enable_passes;
    trace!(?overridden_passes);

    for pass in passes {
        let name = pass.name();

        // Gather information about what we should be doing for this pass
        let overridden =
            overridden_passes.iter().rev().find(|(s, _)| s == &*name).map(|(_name, polarity)| {
                trace!(
                    pass = %name,
                    "{} as requested by flag",
                    if *polarity { "Running" } else { "Not running" },
                );
                *polarity
            });
        let is_enabled = overridden.unwrap_or_else(|| pass.is_enabled(&tcx.sess));
        let new_phase = pass.phase_change();
        let dump_enabled = (is_enabled && pass.is_mir_dump_enabled()) || new_phase.is_some();
        let validate = (validate && is_enabled)
            || new_phase == Some(MirPhase::Runtime(RuntimePhase::Optimized));

        if dump_enabled {
            dump_mir(tcx, body, start_phase, &name, cnt, false);
        }
        if is_enabled {
            pass.run_pass(tcx, body);
        }
        if dump_enabled {
            dump_mir(tcx, body, start_phase, &name, cnt, true);
            cnt += 1;
        }
        if let Some(new_phase) = pass.phase_change() {
            if body.phase >= new_phase {
                panic!("Invalid MIR phase transition from {:?} to {:?}", body.phase, new_phase);
            }

            body.phase = new_phase;
        }
        if validate {
            validate_body(tcx, body, format!("after pass {}", name));
        }
    }
}

pub fn validate_body<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>, when: String) {
    validate::Validator { when, mir_phase: body.phase }.run_pass(tcx, body);
}

pub fn dump_mir<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    phase: MirPhase,
    pass_name: &str,
    cnt: usize,
    is_after: bool,
) {
    let phase_index = phase.phase_index();

    mir::dump_mir(
        tcx,
        Some(&format_args!("{:03}-{:03}", phase_index, cnt)),
        pass_name,
        if is_after { &"after" } else { &"before" },
        body,
        |_, _| Ok(()),
    );
}
