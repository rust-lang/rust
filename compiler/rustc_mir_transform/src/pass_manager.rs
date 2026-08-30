use std::cell::RefCell;
use std::sync::atomic::Ordering;

use rustc_data_structures::fx::{FxHashMap, FxIndexSet};
use rustc_data_structures::hash_map::Entry;
use rustc_middle::mir::{Body, MirDumper, MirPhase, RuntimePhase};
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use tracing::trace;

use crate::lint::lint_body;
use crate::{diagnostics, validate};

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

/// Rules outlining when this pass may be overridden or suppressed.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum PassPolicy {
    /// This pass implements a mandatory lowering step, either to implement parts of the MIR semantics
    /// or to bring MIR into a shape that is easier to deal with for later passes/codegen.
    /// Passes using this cannot be disabled via any means. They must not remove any UB, as they will
    /// run in Miri. They must also come with a comment justifying why they must always run.
    Required,
    /// An optional pass that may be configured by `-Zmir-enable-passes`.
    Optional {
        /// Whether this pass should be enabled by default in this session in the absence of
        /// an explicit `-Zmir-enable-passes` or `#[optimize(none)]`.
        generally_enabled: bool,
        /// Whether this is an optimization pass. `#[optimize(none)]` only disables optimization
        /// passes.
        /// A pass may be optional without being an optimization pass,
        /// e.g. if it just adds extra debug checks that one can turn off.
        optimization: bool,
    },
}

impl PassPolicy {
    fn and_enabled(self, enabled: bool) -> Self {
        match self {
            PassPolicy::Required => PassPolicy::Required,
            PassPolicy::Optional { generally_enabled: enabled_by_default, optimization } => {
                PassPolicy::Optional {
                    generally_enabled: enabled_by_default && enabled,
                    optimization,
                }
            }
        }
    }

    /// Create a [`PassPolicy::Optional`] that is not an optimization,
    /// enabled by default under the given condition.
    pub(crate) fn optional_non_optimization(condition: bool) -> Self {
        Self::Optional { generally_enabled: condition, optimization: false }
    }

    /// Create a [`PassPolicy::Optional`] optimization, enabled by default under the given condition.
    pub(crate) fn optimization(condition: bool) -> Self {
        Self::Optional { generally_enabled: condition, optimization: true }
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

    /// Describes how this pass is enabled and which mechanisms may disable it.
    fn policy(&self, sess: &Session) -> PassPolicy;

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>);

    fn is_mir_dump_enabled(&self) -> bool {
        true
    }
}

/// Just like `MirPass`, except it cannot mutate `Body`, and MIR dumping is
/// disabled (via the `Lint` adapter).
pub(super) trait MirLint<'tcx> {
    fn name(&self) -> &'static str {
        const { simplify_pass_type_name(std::any::type_name::<Self>()) }
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

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        self.0.run_lint(tcx, body)
    }

    fn is_mir_dump_enabled(&self) -> bool {
        false
    }

    fn policy(&self, _sess: &Session) -> PassPolicy {
        PassPolicy::optional_non_optimization(true)
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

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        self.1.run_pass(tcx, body)
    }

    fn policy(&self, sess: &Session) -> PassPolicy {
        self.1.policy(sess).and_enabled(sess.mir_opt_level() >= self.0 as usize)
    }
}

/// Whether to allow [optimization passes].
///
/// [optimization passes]: PassPolicy::Optional::optimization
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Optimizations {
    /// The current function has `#[optimize(none)]`.
    Suppressed,
    /// Normal optimizations may run.
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

pub(super) fn should_run_pass<'tcx, P>(
    tcx: TyCtxt<'tcx>,
    pass: &P,
    optimizations: Optimizations,
) -> bool
where
    P: MirPass<'tcx> + ?Sized,
{
    let name = pass.name();
    let pass_override = || {
        tcx.sess
            .opts
            .unstable_opts
            .mir_enable_passes
            .iter()
            .rev()
            .find_map(|(name_, polarity)| if name == name_ { Some(*polarity) } else { None })
    };

    match pass.policy(tcx.sess) {
        PassPolicy::Required => true,
        PassPolicy::Optional { generally_enabled: enabled_by_default, optimization } => {
            if let Some(o) = pass_override() {
                trace!(
                    pass = %name,
                    "{} as requested by flag",
                    if o { "Running" } else { "Not running" }
                );
                o
            } else if optimization && optimizations == Optimizations::Suppressed {
                trace!(pass = %name, "Not running as requested by `#[optimize(none)]`");
                false
            } else {
                enabled_by_default
            }
        }
    }
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

    let named_passes: FxIndexSet<_> =
        overridden_passes.iter().map(|(name, _)| name.as_str()).collect();

    let mut unknown_found = false;
    for &name in named_passes.difference(&*crate::PASS_NAMES) {
        tcx.dcx().emit_warn(diagnostics::UnknownPassName { name });
        unknown_found = true;
    }

    if unknown_found {
        let mut valid_pass_names = crate::PASS_NAMES.iter().copied().collect::<Vec<_>>();
        valid_pass_names.sort();
        tcx.dcx().emit_note(diagnostics::ValidPassNames { valid_passes: valid_pass_names.into() });
    }

    // Verify that no passes are missing from the `declare_passes` invocation
    #[cfg(debug_assertions)]
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

        let def_id = body.source.def_id();
        let optimizations = if tcx.def_kind(def_id).has_codegen_attrs()
            && tcx.codegen_fn_attrs(def_id).optimize.do_not_optimize()
        {
            Optimizations::Suppressed
        } else {
            Optimizations::Allowed
        };

        for pass in passes {
            let pass_name = pass.name();

            if !should_run_pass(tcx, *pass, optimizations) {
                continue;
            };

            if is_optimization_stage(body, phase_change, optimizations)
                && let Some(limit) = &tcx.sess.opts.unstable_opts.mir_opt_bisect_limit
                && limited_by_opt_bisect(
                    tcx,
                    tcx.def_path_debug_str(body.source.def_id()),
                    *limit,
                    *pass,
                )
            {
                continue;
            }

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

fn is_optimization_stage(
    body: &Body<'_>,
    phase_change: Option<MirPhase>,
    optimizations: Optimizations,
) -> bool {
    optimizations == Optimizations::Allowed
        && body.phase == MirPhase::Runtime(RuntimePhase::PostCleanup)
        && phase_change == Some(MirPhase::Runtime(RuntimePhase::Optimized))
}

fn limited_by_opt_bisect<'tcx, P>(
    tcx: TyCtxt<'tcx>,
    def_path: String,
    limit: usize,
    pass: &P,
) -> bool
where
    P: MirPass<'tcx> + ?Sized,
{
    let current_opt_bisect_count =
        tcx.sess.mir_opt_bisect_eval_count.fetch_add(1, Ordering::Relaxed);

    let can_run = current_opt_bisect_count < limit;

    if can_run {
        eprintln!(
            "BISECT: running pass ({}) {} on {}",
            current_opt_bisect_count + 1,
            pass.name(),
            def_path
        );
    } else {
        eprintln!(
            "BISECT: NOT running pass ({}) {} on {}",
            current_opt_bisect_count + 1,
            pass.name(),
            def_path
        );
    }

    !can_run
}
