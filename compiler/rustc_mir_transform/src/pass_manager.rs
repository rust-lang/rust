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
}

/// Run the sequence of passes without validating the MIR after each pass. The MIR is still
/// validated at the end.
pub fn run_passes_no_validate<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    passes: &[&dyn MirPass<'tcx>],
    phase_change: Option<MirPhase>,
) {
    run_passes_inner(tcx, body, passes, phase_change, false);
}

/// The optional `phase_change` is applied after executing all the passes, if present
pub fn run_passes<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    passes: &[&dyn MirPass<'tcx>],
    phase_change: Option<MirPhase>,
) {
    run_passes_inner(tcx, body, passes, phase_change, true);
}

fn run_passes_inner<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    passes: &[&dyn MirPass<'tcx>],
    phase_change: Option<MirPhase>,
    validate_each: bool,
) {
    let validate = validate_each & tcx.sess.opts.unstable_opts.validate_mir;
    let overridden_passes = &tcx.sess.opts.unstable_opts.mir_enable_passes;
    trace!(?overridden_passes);

    for pass in passes {
        let name = pass.name();

        let overridden =
            overridden_passes.iter().rev().find(|(s, _)| s == &*name).map(|(_name, polarity)| {
                trace!(
                    pass = %name,
                    "{} as requested by flag",
                    if *polarity { "Running" } else { "Not running" },
                );
                *polarity
            });
        if !overridden.unwrap_or_else(|| pass.is_enabled(&tcx.sess)) {
            continue;
        }

        let dump_enabled = pass.is_mir_dump_enabled();

        if dump_enabled {
            dump_mir_for_pass(tcx, body, &name, false);
        }
        if validate {
            validate_body(tcx, body, format!("before pass {}", name));
        }

        pass.run_pass(tcx, body);

        if dump_enabled {
            dump_mir_for_pass(tcx, body, &name, true);
        }
        if validate {
            validate_body(tcx, body, format!("after pass {}", name));
        }

        body.pass_count += 1;
    }

    if let Some(new_phase) = phase_change {
        if body.phase >= new_phase {
            panic!("Invalid MIR phase transition from {:?} to {:?}", body.phase, new_phase);
        }

        body.phase = new_phase;

        dump_mir_for_phase_change(tcx, body);
        if validate || new_phase == MirPhase::Runtime(RuntimePhase::Optimized) {
            validate_body(tcx, body, format!("after phase change to {}", new_phase));
        }

        body.pass_count = 1;
    }
}

pub fn validate_body<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>, when: String) {
    validate::Validator { when, mir_phase: body.phase }.run_pass(tcx, body);
}

pub fn dump_mir_for_pass<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    pass_name: &str,
    is_after: bool,
) {
    let phase_index = body.phase.phase_index();

    mir::dump_mir(
        tcx,
        Some(&format_args!("{:03}-{:03}", phase_index, body.pass_count)),
        pass_name,
        if is_after { &"after" } else { &"before" },
        body,
        |_, _| Ok(()),
    );
}

pub fn dump_mir_for_phase_change<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>) {
    let phase_index = body.phase.phase_index();

    mir::dump_mir(
        tcx,
        Some(&format_args!("{:03}-000", phase_index)),
        &format!("{}", body.phase),
        &"after",
        body,
        |_, _| Ok(()),
    )
}
