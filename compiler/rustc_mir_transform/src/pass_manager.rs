use rustc_middle::mir::{self, Body, MirPhase, RuntimePhase};
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;

use crate::{lint::lint_body, validate, MirPass};

/// Just like `MirPass`, except it cannot mutate `Body`.
pub trait MirLint<'tcx> {
    fn name(&self) -> &'static str {
        // FIXME Simplify the implementation once more `str` methods get const-stable.
        // See copypaste in `MirPass`
        const {
            let name = std::any::type_name::<Self>();
            rustc_middle::util::common::c_name(name)
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
    fn name(&self) -> &'static str {
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
    fn name(&self) -> &'static str {
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

pub fn should_run_pass<'tcx, P>(tcx: TyCtxt<'tcx>, pass: &P) -> bool
where
    P: MirPass<'tcx> + ?Sized,
{
    let name = pass.name();

    let overridden_passes = &tcx.sess.opts.unstable_opts.mir_enable_passes;
    let overridden =
        overridden_passes.iter().rev().find(|(s, _)| s == &*name).map(|(_name, polarity)| {
            trace!(
                pass = %name,
                "{} as requested by flag",
                if *polarity { "Running" } else { "Not running" },
            );
            *polarity
        });
    overridden.unwrap_or_else(|| pass.is_enabled(tcx.sess))
}

fn run_passes_inner<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    passes: &[&dyn MirPass<'tcx>],
    phase_change: Option<MirPhase>,
    validate_each: bool,
) {
    let overridden_passes = &tcx.sess.opts.unstable_opts.mir_enable_passes;
    trace!(?overridden_passes);

    let prof_arg = tcx.sess.prof.enabled().then(|| format!("{:?}", body.source.def_id()));

    if !body.should_skip() {
        let validate = validate_each & tcx.sess.opts.unstable_opts.validate_mir;
        let lint = tcx.sess.opts.unstable_opts.lint_mir;

        for pass in passes {
            let name = pass.name();

            if !should_run_pass(tcx, *pass) {
                continue;
            };

            let dump_enabled = pass.is_mir_dump_enabled();

            if dump_enabled {
                dump_mir_for_pass(tcx, body, name, false);
            }

            if let Some(prof_arg) = &prof_arg {
                tcx.sess
                    .prof
                    .generic_activity_with_arg(pass.profiler_name(), &**prof_arg)
                    .run(|| pass.run_pass(tcx, body));
            } else {
                pass.run_pass(tcx, body);
            }

            if dump_enabled {
                dump_mir_for_pass(tcx, body, name, true);
            }
            if validate {
                validate_body(tcx, body, format!("after pass {name}"));
            }
            if lint {
                lint_body(tcx, body, format!("after pass {name}"));
            }

            body.pass_count += 1;
        }
    }

    if let Some(new_phase) = phase_change {
        if body.phase >= new_phase {
            panic!("Invalid MIR phase transition from {:?} to {:?}", body.phase, new_phase);
        }

        body.phase = new_phase;
        body.pass_count = 0;

        dump_mir_for_phase_change(tcx, body);

        let validate =
            (validate_each & tcx.sess.opts.unstable_opts.validate_mir & !body.should_skip())
                || new_phase == MirPhase::Runtime(RuntimePhase::Optimized);
        let lint = tcx.sess.opts.unstable_opts.lint_mir & !body.should_skip();
        if validate {
            validate_body(tcx, body, format!("after phase change to {}", new_phase.name()));
        }
        if lint {
            lint_body(tcx, body, format!("after phase change to {}", new_phase.name()));
        }

        body.pass_count = 1;
    }

    if let Some(coroutine) = body.coroutine.as_mut() {
        if let Some(by_move_body) = coroutine.by_move_body.as_mut() {
            run_passes_inner(tcx, by_move_body, passes, phase_change, validate_each);
        }
        if let Some(by_mut_body) = coroutine.by_mut_body.as_mut() {
            run_passes_inner(tcx, by_mut_body, passes, phase_change, validate_each);
        }
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
    mir::dump_mir(
        tcx,
        true,
        pass_name,
        if is_after { &"after" } else { &"before" },
        body,
        |_, _| Ok(()),
    );
}

pub fn dump_mir_for_phase_change<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>) {
    assert_eq!(body.pass_count, 0);
    mir::dump_mir(tcx, true, body.phase.name(), &"after", body, |_, _| Ok(()))
}
