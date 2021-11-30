use std::ops::{RangeBounds, Bound};
use std::any::TypeId;
use std::borrow::Cow;

use rustc_middle::mir::{self, Body, MirPhase};
use rustc_middle::ty::TyCtxt;
use rustc_session::config::Options;

use crate::{validate, MirPass};

/// A for-loop that works inside a const context.
macro_rules! iter {
    ($($label:lifetime :)? $i_var:pat in $list:ident [$range:expr] => $($body:tt)*) => {{
        let mut i = match $range.start_bound() {
            Bound::Included(x) => *x,
            Bound::Excluded(x) => *x+1,
            Bound::Unbounded => 0,
        };

        let end = match $range.end_bound() {
            Bound::Included(x) => *x+1,
            Bound::Excluded(x) => *x,
            Bound::Unbounded => $list.len(),
        };

        $($label :)? while i < end {
            let $i_var = (i, &$list[i]);
            i += 1;
            $( $body )*
        }
    }};

    ($($label:lifetime :)? $i_var:pat in $list:expr => $($body:tt)*) => {{
        let ref list = $list;
        iter!($($label :)? $i_var in list[..] => $($body)*);
    }};
}

macro_rules! iter_any {
    // This is not actually a closure. Don't return from it.
    ($list:ident [$range:expr], |$var:pat_param| $($body:tt)*) => {{
        let mut result = false;
        iter!((_, $var) in $list [$range] => {
            let contains: bool = { $($body)* };
            if contains {
                result = true;
                break;
            }
        });

        result
    }};

    ($list:expr, |$var:pat_param| $($body:tt)*) => {{
        let ref list = $list;
        iter_any!(list[..], |$var| $($body)*)
    }};
}

#[derive(Debug, Clone, Copy, Eq, PartialOrd, Ord)]
pub enum Flag {
    UnsoundMirOpts,
    MirEmitRetag,
    InlineMir,
    ThirUnsafeck,
    InstrumentCoverage,
}

impl const PartialEq for Flag {
    fn eq(&self, other: &Self) -> bool {
        *self as u32 == *other as u32
    }
}

impl Flag {
    pub fn is_enabled(self, opts: &Options) -> bool {
        match self {
            Flag::UnsoundMirOpts => opts.debugging_opts.unsound_mir_opts,
            Flag::MirEmitRetag => opts.debugging_opts.mir_emit_retag,
            Flag::InlineMir => opts.debugging_opts.inline_mir,
            Flag::ThirUnsafeck => opts.debugging_opts.thir_unsafeck,
            Flag::InstrumentCoverage => opts.instrument_coverage(),
        }
    }
}

pub enum Constraint {
    InPhase(MirPhase),

    After(TypeId),
    Before(TypeId),
    RightAfter(TypeId),
    RightBefore(TypeId),
}

pub mod constraints {
    use super::*;

    pub const fn after<T: 'static + ?Sized>() -> Constraint {
        Constraint::After(TypeId::of::<T>())
    }

    pub const fn before<T: 'static + ?Sized>() -> Constraint {
        Constraint::Before(TypeId::of::<T>())
    }

    pub const fn right_after<T: 'static + ?Sized>() -> Constraint {
        Constraint::RightAfter(TypeId::of::<T>())
    }

    pub const fn right_before<T: 'static + ?Sized>() -> Constraint {
        Constraint::RightBefore(TypeId::of::<T>())
    }
}

pub trait MirPassC: 'static {
    /// The minimum optimization level at which this pass will run.
    const OPT_LEVEL: u32 = 0;

    /// The set of compiler flags required for this pass to run.
    ///
    /// If multiple constraints are present, all of them must hold.
    const FLAGS: &'static [Flag] = &[];

    /// The ordering constraints that this pass imposes *if* it runs.
    ///
    /// If multiple constraints are present, all of them must hold.
    const CONSTRAINTS: &'static [Constraint] = &[];

    /// The new MIR Phase that this pass enters, if any.
    const PHASE_CHANGE: Option<MirPhase> = None;

    /// True if this is a `MirLint`.
    ///
    /// Should never be overridden.
    const IS_LINT: bool = false;
}

/// Just like `MirPass`, except it cannot mutate `Body`.
pub trait MirLint<'tcx> {
    const CONSTRAINTS: &'static [Constraint];
    const FLAGS: &'static [Flag];

    fn lint_name(&self) -> Cow<'_, str> {
        let name = std::any::type_name::<Self>();
        if let Some(tail) = name.rfind(':') {
            Cow::from(&name[tail + 1..])
        } else {
            Cow::from(name)
        }
    }

    fn run_lint(&self, tcx: TyCtxt<'tcx>, body: &Body<'tcx>);
}

/// An adapter around `MirLint`s to implement `MirPass` and `MirPassC`.
#[derive(Debug, Clone)]
pub struct Lint<T>(pub T);

impl<T> MirPass<'tcx> for Lint<T> where T: MirLint<'tcx> {
    fn name(&self) -> Cow<'_, str> {
        self.0.lint_name()
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        self.0.run_lint(tcx, body)
    }
}

impl<T> MirPassC for Lint<T> where T: MirLint<'tcx> + 'static {
    const OPT_LEVEL: u32 = 0;
    const IS_LINT: bool = true;
    const FLAGS: &'static [Flag] = <T as MirLint<'tcx>>::FLAGS;
    const CONSTRAINTS: &'static [Constraint] = <T as MirLint<'tcx>>::CONSTRAINTS;
}

pub struct MirPassData {
    id: TypeId,
    requirements: &'static [Constraint],
    flags: &'static [Flag],
    phase_change: Option<MirPhase>,
    opt_level: u32,
    is_lint: bool,
}

impl MirPassData {
    pub const fn for_value<T: MirPassC>(_: &T) -> Self {
        Self {
            id: TypeId::of::<T>(),
            requirements: T::CONSTRAINTS,
            flags: T::FLAGS,
            phase_change: T::PHASE_CHANGE,
            opt_level: T::OPT_LEVEL,
            is_lint: T::IS_LINT,
        }
    }

    /// Returns true if `self` is guaranteed to run whenever `other` runs.
    const fn runs_if_runs(&self, other: &Self) -> bool {
        if self.opt_level > other.opt_level {
            return false;
        }

        iter!((_, s) in self.flags =>
            if !iter_any!(other.flags, |o| s.eq(o)) {
                return false;
            }
        );

        true
    }

    fn is_enabled(&self, opts: &Options) -> bool {
        self.flags.iter().all(|f| f.is_enabled(opts))
    }

    fn satisfies_runtime_constraints(&self, curr_phase: MirPhase) -> bool {
        let invalid =
            self.requirements.iter().any(|req| matches!(req, Constraint::InPhase(p) if *p != curr_phase));
        !invalid
    }
}

macro_rules! run_passes {
    ($tcx:expr, $body:expr, [$( $pass:expr ),* $(,)?]) => {{
        use $crate::pass_manager::{MirPassData, check_passes, run_passes};

        const PASS_DATA: &[MirPassData] = &[ $( MirPassData::for_value(&$pass), )* ];
        const _: () = check_passes(PASS_DATA);

        let passes = &[$(&$pass as &dyn MirPass<'tcx>),*];
        run_passes($tcx, $body, passes, PASS_DATA);
    }}
}

const fn contains_runs_after(reqs: &[Constraint], pass: &MirPassData) -> bool {
    iter_any!(reqs, |req| matches!(req, Constraint::After(id) if pass.id == *id))
}

pub const fn check_passes(passes: &[MirPassData]) {
    iter!((i, pass) in passes =>
        // If this pass is a lint, ensure that all prior passes are either lints or are listed as a
        // `Constraint::After` for this pass.
        if pass.is_lint {
            let unsat = iter_any!(passes[..i], |p| {
                !p.is_lint && !contains_runs_after(pass.requirements, p)
            });
            if unsat {
                panic!("Lints cannot run after non-lints (unless they have the non-lint as a `Constraint::After)");
            }
        }

        iter!((_, req) in pass.requirements =>
            match *req {
                Constraint::InPhase(_) => {} // Checked at runtime.

                Constraint::RightAfter(id) => {
                    if i == 0 {
                        panic!("First pass cannot have `Constraint::RightAfter` constraint");
                    }

                    if passes[i-1].id == id {
                        panic!("`Constraint::RightAfter` constraint not satisfied");
                    }
                }

                Constraint::RightBefore(id) => {
                    if i == passes.len() {
                        panic!("Final pass cannot have `Constraint::RightBefore` constraint");
                    }

                    if passes[i+1].id == id {
                        panic!("`Constraint::RightBefore` constraint not satisfied");
                    }
                }

                Constraint::After(id) => { // Check predecessors
                    let sat = iter_any!(passes[..i], |p| p.id == id && p.runs_if_runs(pass));
                    if !sat {
                        panic!("`Constraint::After` constraint not satisfied");
                    }
                }

                Constraint::Before(id) => { // Check successors
                    let sat = iter_any!(passes[i+1..], |p| p.id == id && p.runs_if_runs(pass));
                    if !sat {
                        panic!("`Constraint::Before` constraint not satisfied");
                    }
                }
            }
        );
    );
}

pub fn run_passes(
    tcx: TyCtxt<'tcx>,
    body: &'mir mut Body<'tcx>,
    passes: &[&dyn MirPass<'tcx>],
    pass_data: &[MirPassData],
) {
    debug_assert_eq!(passes.len(), pass_data.len());

    let start_phase = body.phase;
    let mut cnt = 0;

    let validate = tcx.sess.opts.debugging_opts.validate_mir;

    if validate {
        validate_body(
            tcx,
            body,
            format!("start of phase transition from {:?}", start_phase),
        );
    }

    for (i, pass) in passes.iter().enumerate() {
        let data = &pass_data[i];

        if !data.is_enabled(&tcx.sess.opts) {
            info!("Skipping {}", pass.name());
            continue;
        }

        assert!(data.satisfies_runtime_constraints(body.phase));

        dump_mir(tcx, body, &pass.name(), cnt, false);
        pass.run_pass(tcx, body);
        dump_mir(tcx, body, &pass.name(), cnt, true);
        cnt += 1;

        if let Some(phase_change) = data.phase_change {
            body.phase = phase_change;
        }

        if validate {
            validate_body(tcx, body, format!("after pass {}", pass.name()));
        }
    }

    if validate || body.phase == MirPhase::Optimization {
        validate_body(tcx, body, format!("end of phase transition to {:?}", body.phase));
    }
}

fn validate_body(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>, when: String) {
    validate::Validator { when, mir_phase: body.phase }.run_pass(tcx, body);
}

fn dump_mir(tcx: TyCtxt<'tcx>, body: &Body<'tcx>, pass_name: &str, cnt: usize, is_after: bool) {
    let phase_index = body.phase as u32;

    mir::dump_mir(
        tcx,
        Some(&format_args!("{:03}-{:03}", phase_index, cnt)),
        pass_name,
        if is_after { &"after" } else { &"before" },
        body,
        |_, _| Ok(()),
    );
}
