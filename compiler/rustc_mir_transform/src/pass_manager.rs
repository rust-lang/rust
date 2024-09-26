use std::cell::RefCell;
use std::collections::hash_map::Entry;

use rustc_data_structures::fx::FxHashMap;
use rustc_middle::mir::{self, Body, MirPhase, RuntimePhase};
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use tracing::trace;

use crate::lint::lint_body;
use crate::validate;

thread_local! {
    static PASS_NAMES: RefCell<FxHashMap<&'static str, &'static str>> = {
        RefCell::new(FxHashMap::default())
    };
}

/// Converts a MIR pass name into a snake case form to match the profiling naming style.
fn to_profiler_name(type_name: &'static str) -> &'static str {
    PASS_NAMES.with(|names| match names.borrow_mut().entry(type_name) {
        Entry::Occupied(e) => *e.get(),
        Entry::Vacant(e) => {
            let snake_case: String = type_name
                .chars()
                .flat_map(|c| {
                    if c.is_ascii_uppercase() {
                        vec!['_', c.to_ascii_lowercase()]
                    } else if c == '-' {
                        vec!['_']
                    } else {
                        vec![c]
                    }
                })
                .collect();
            let result = &*String::leak(format!("mir_pass{}", snake_case));
            e.insert(result);
            result
        }
    })
}

// const wrapper for `if let Some((_, tail)) = name.rsplit_once(':') { tail } else { name }`
const fn c_name(name: &'static str) -> &'static str {
    // FIXME(const-hack) Simplify the implementation once more `str` methods get const-stable.
    // and inline into call site
    let bytes = name.as_bytes();
    let mut i = bytes.len();
    while i > 0 && bytes[i - 1] != b':' {
        i = i - 1;
    }
    let (_, bytes) = bytes.split_at(i);
    match std::str::from_utf8(bytes) {
        Ok(name) => name,
        Err(_) => name,
    }
}

/// A streamlined trait that you can implement to create a pass; the
/// pass will be named after the type, and it will consist of a main
/// loop that goes over each available MIR and applies `run_pass`.
pub(super) trait MirPass<'tcx> {
    fn name(&self) -> &'static str {
        // FIXME(const-hack) Simplify the implementation once more `str` methods get const-stable.
        // See copypaste in `MirLint`
        const {
            let name = std::any::type_name::<Self>();
            c_name(name)
        }
    }

    fn profiler_name(&self) -> &'static str {
        to_profiler_name(self.name())
    }

    /// Returns `true` if this pass is enabled with the current combination of compiler flags.
    fn is_enabled(&self, _sess: &Session) -> bool {
        true
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>);

    fn is_mir_dump_enabled(&self) -> bool {
        true
    }
}

/// Just like `MirPass`, except it cannot mutate `Body`, and MIR dumping is
/// disabled (via the `Lint` adapter).
pub(super) trait MirLint<'tcx> {
    fn name(&self) -> &'static str {
        // FIXME(const-hack) Simplify the implementation once more `str` methods get const-stable.
        // See copypaste in `MirPass`
        const {
            let name = std::any::type_name::<Self>();
            c_name(name)
        }
    }

    fn is_enabled(&self, _sess: &Session) -> bool {
        true
    }

    fn run_lint(&self, tcx: TyCtxt<'tcx>, body: &Body<'tcx>);
}

/// An adapter for `MirLint`s that implements `MirPass`.
#[derive(Debug, Clone)]
pub(super) struct Lint<T>(pub T);

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

pub(super) struct WithMinOptLevel<T>(pub u32, pub T);

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
pub(super) fn run_passes_no_validate<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    passes: &[&dyn MirPass<'tcx>],
    phase_change: Option<MirPhase>,
) {
    run_passes_inner(tcx, body, passes, phase_change, false);
}

/// The optional `phase_change` is applied after executing all the passes, if present
pub(super) fn run_passes<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    passes: &[&dyn MirPass<'tcx>],
    phase_change: Option<MirPhase>,
) {
    run_passes_inner(tcx, body, passes, phase_change, true);
}

pub(super) fn should_run_pass<'tcx, P>(tcx: TyCtxt<'tcx>, pass: &P) -> bool
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
}

pub(super) fn validate_body<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>, when: String) {
    validate::Validator { when, mir_phase: body.phase }.run_pass(tcx, body);
}

fn dump_mir_for_pass<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>, pass_name: &str, is_after: bool) {
    mir::dump_mir(
        tcx,
        true,
        pass_name,
        if is_after { &"after" } else { &"before" },
        body,
        |_, _| Ok(()),
    );
}

pub(super) fn dump_mir_for_phase_change<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>) {
    assert_eq!(body.pass_count, 0);
    mir::dump_mir(tcx, true, body.phase.name(), &"after", body, |_, _| Ok(()))
}
