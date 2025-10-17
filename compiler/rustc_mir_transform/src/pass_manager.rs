use std::cell::RefCell;
use std::collections::hash_map::Entry;

use rustc_data_structures::fx::{FxHashMap, FxIndexSet};
use rustc_middle::mir::{Body, MirDumper, MirPhase, RuntimePhase};
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use tracing::trace;

use crate::lint::lint_body;
use crate::{errors, validate};

thread_local! {
    /// Maps MIR pass names to a snake case form to match profiling naming style
    static PASS_TO_PROFILER_NAMES: RefCell<FxHashMap<&'static str, &'static str>> = {
        RefCell::new(FxHashMap::default())
    };
}

/// Converts a MIR pass name into a snake case form to match the profiling naming style.
fn to_profiler_name(type_name: &'static str) -> &'static str {
    PASS_TO_PROFILER_NAMES.with(|names| match names.borrow_mut().entry(type_name) {
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

// A function that simplifies a pass's type_name. E.g. `Baz`, `Baz<'_>`,
// `foo::bar::Baz`, and `foo::bar::Baz<'a, 'b>` all become `Baz`.
//
// It's `const` for perf reasons: it's called a lot, and doing the string
// operations at runtime causes a non-trivial slowdown. If
// `split_once`/`rsplit_once` become `const` its body could be simplified to
// this:
// ```ignore (fragment)
// let name = if let Some((_, tail)) = name.rsplit_once(':') { tail } else { name };
// let name = if let Some((head, _)) = name.split_once('<') { head } else { name };
// name
// ```
const fn simplify_pass_type_name(name: &'static str) -> &'static str {
    // FIXME(const-hack) Simplify the implementation once more `str` methods get const-stable.

    // Work backwards from the end. If a ':' is hit, strip it and everything before it.
    let bytes = name.as_bytes();
    let mut i = bytes.len();
    while i > 0 && bytes[i - 1] != b':' {
        i -= 1;
    }
    let (_, bytes) = bytes.split_at(i);

    // Work forwards from the start of what's left. If a '<' is hit, strip it and everything after
    // it.
    let mut i = 0;
    while i < bytes.len() && bytes[i] != b'<' {
        i += 1;
    }
    let (bytes, _) = bytes.split_at(i);

    match std::str::from_utf8(bytes) {
        Ok(name) => name,
        Err(_) => panic!(),
    }
}

/// A streamlined trait that you can implement to create a pass; the
/// pass will be named after the type, and it will consist of a main
/// loop that goes over each available MIR and applies `run_pass`.
pub(super) trait MirPass<'tcx> {
    fn name(&self) -> &'static str {
        const { simplify_pass_type_name(std::any::type_name::<Self>()) }
    }

    fn profiler_name(&self) -> &'static str {
        to_profiler_name(self.name())
    }

    /// Returns `true` if this pass is enabled with the current combination of compiler flags.
    fn is_enabled(&self, _sess: &Session) -> bool {
        true
    }

    /// Returns `true` if this pass can be overridden by `-Zenable-mir-passes`. This should be
    /// true for basically every pass other than those that are necessary for correctness.
    fn can_be_overridden(&self) -> bool {
        true
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>);

    fn is_mir_dump_enabled(&self) -> bool {
        true
    }

    /// Returns `true` if this pass must be run (i.e. it is required for soundness).
    /// For passes which are strictly optimizations, this should return `false`.
    /// If this is `false`, `#[optimize(none)]` will disable the pass.
    fn is_required(&self) -> bool;
}

/// Just like `MirPass`, except it cannot mutate `Body`, and MIR dumping is
/// disabled (via the `Lint` adapter).
pub(super) trait MirLint<'tcx> {
    fn name(&self) -> &'static str {
        const { simplify_pass_type_name(std::any::type_name::<Self>()) }
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

    fn is_required(&self) -> bool {
        true
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

    fn is_required(&self) -> bool {
        self.1.is_required()
    }
}

/// Whether to allow non-[required] optimizations
///
/// [required]: MirPass::is_required
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Optimizations {
    Suppressed,
    Allowed,
}

/// Run the sequence of passes without validating the MIR after each pass. The MIR is still
/// validated at the end.
pub(super) fn run_passes_no_validate<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    passes: &[&dyn MirPass<'tcx>],
    phase_change: Option<MirPhase>,
) {
    run_passes_inner(tcx, body, passes, phase_change, false, Optimizations::Allowed);
}

/// The optional `phase_change` is applied after executing all the passes, if present
pub(super) fn run_passes<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    passes: &[&dyn MirPass<'tcx>],
    phase_change: Option<MirPhase>,
    optimizations: Optimizations,
) {
    run_passes_inner(tcx, body, passes, phase_change, true, optimizations);
}

pub(super) fn should_run_pass<'tcx, P>(
    tcx: TyCtxt<'tcx>,
    pass: &P,
    optimizations: Optimizations,
) -> bool
where
    P: MirPass<'tcx> + ?Sized,
{
    let name = pass.name();

    if !pass.can_be_overridden() {
        return pass.is_enabled(tcx.sess);
    }

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
    let suppressed = !pass.is_required() && matches!(optimizations, Optimizations::Suppressed);
    overridden.unwrap_or_else(|| !suppressed && pass.is_enabled(tcx.sess))
}

fn run_passes_inner<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    passes: &[&dyn MirPass<'tcx>],
    phase_change: Option<MirPhase>,
    validate_each: bool,
    optimizations: Optimizations,
) {
    let overridden_passes = &tcx.sess.opts.unstable_opts.mir_enable_passes;
    trace!(?overridden_passes);

    let named_passes: FxIndexSet<_> =
        overridden_passes.iter().map(|(name, _)| name.as_str()).collect();

    for &name in named_passes.difference(&*crate::PASS_NAMES) {
        tcx.dcx().emit_warn(errors::UnknownPassName { name });
    }

    // Verify that no passes are missing from the `declare_passes` invocation
    #[cfg(debug_assertions)]
    #[allow(rustc::diagnostic_outside_of_impl)]
    #[allow(rustc::untranslatable_diagnostic)]
    {
        let used_passes: FxIndexSet<_> = passes.iter().map(|p| p.name()).collect();

        let undeclared = used_passes.difference(&*crate::PASS_NAMES).collect::<Vec<_>>();
        if let Some((name, rest)) = undeclared.split_first() {
            let mut err =
                tcx.dcx().struct_bug(format!("pass `{name}` is not declared in `PASS_NAMES`"));
            for name in rest {
                err.note(format!("pass `{name}` is also not declared in `PASS_NAMES`"));
            }
            err.emit();
        }
    }

    let prof_arg = tcx.sess.prof.enabled().then(|| format!("{:?}", body.source.def_id()));

    if !body.should_skip() {
        let validate = validate_each & tcx.sess.opts.unstable_opts.validate_mir;
        let lint = tcx.sess.opts.unstable_opts.lint_mir;

        for pass in passes {
            let pass_name = pass.name();

            if !should_run_pass(tcx, *pass, optimizations) {
                continue;
            };

            let dumper = if pass.is_mir_dump_enabled()
                && let Some(dumper) = MirDumper::new(tcx, pass_name, body)
            {
                Some(dumper.set_show_pass_num().set_disambiguator(&"before"))
            } else {
                None
            };

            if let Some(dumper) = dumper.as_ref() {
                dumper.dump_mir(body);
            }

            if let Some(prof_arg) = &prof_arg {
                tcx.sess
                    .prof
                    .generic_activity_with_arg(pass.profiler_name(), &**prof_arg)
                    .run(|| pass.run_pass(tcx, body));
            } else {
                pass.run_pass(tcx, body);
            }

            if let Some(dumper) = dumper {
                dumper.set_disambiguator(&"after").dump_mir(body);
            }

            if validate {
                validate_body(tcx, body, format!("after pass {pass_name}"));
            }
            if lint {
                lint_body(tcx, body, format!("after pass {pass_name}"));
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
    validate::Validator { when }.run_pass(tcx, body);
}

pub(super) fn dump_mir_for_phase_change<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>) {
    assert_eq!(body.pass_count, 0);
    if let Some(dumper) = MirDumper::new(tcx, body.phase.name(), body) {
        dumper.set_show_pass_num().set_disambiguator(&"after").dump_mir(body)
    }
}
