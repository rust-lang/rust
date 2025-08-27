use clippy_config::Conf;
use clippy_utils::diagnostics::{span_lint_and_help, span_lint_and_sugg};
use clippy_utils::is_from_proc_macro;
use clippy_utils::msrvs::Msrv;
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::{Block, Body, HirId, Path, PathSegment, StabilityLevel, StableSince};
use rustc_lint::{LateContext, LateLintPass, Lint, LintContext};
use rustc_session::impl_lint_pass;
use rustc_span::symbol::kw;
use rustc_span::{Span, sym};

declare_clippy_lint! {
    /// ### What it does
    /// Finds items imported through `std` when available through `core`.
    ///
    /// ### Why restrict this?
    /// Crates which have `no_std` compatibility may wish to ensure types are imported from core to ensure
    /// disabling `std` does not cause the crate to fail to compile. This lint is also useful for crates
    /// migrating to become `no_std` compatible.
    ///
    /// ### Example
    /// ```no_run
    /// use std::hash::Hasher;
    /// ```
    /// Use instead:
    /// ```no_run
    /// use core::hash::Hasher;
    /// ```
    #[clippy::version = "1.64.0"]
    pub STD_INSTEAD_OF_CORE,
    restriction,
    "type is imported from std when available in core"
}

declare_clippy_lint! {
    /// ### What it does
    /// Finds items imported through `std` when available through `alloc`.
    ///
    /// ### Why restrict this?
    /// Crates which have `no_std` compatibility and require alloc may wish to ensure types are imported from
    /// alloc to ensure disabling `std` does not cause the crate to fail to compile. This lint is also useful
    /// for crates migrating to become `no_std` compatible.
    ///
    /// ### Example
    /// ```no_run
    /// use std::vec::Vec;
    /// ```
    /// Use instead:
    /// ```no_run
    /// # extern crate alloc;
    /// use alloc::vec::Vec;
    /// ```
    #[clippy::version = "1.64.0"]
    pub STD_INSTEAD_OF_ALLOC,
    restriction,
    "type is imported from std when available in alloc"
}

declare_clippy_lint! {
    /// ### What it does
    /// Finds items imported through `alloc` when available through `core`.
    ///
    /// ### Why restrict this?
    /// Crates which have `no_std` compatibility and may optionally require alloc may wish to ensure types are
    /// imported from core to ensure disabling `alloc` does not cause the crate to fail to compile. This lint
    /// is also useful for crates migrating to become `no_std` compatible.
    ///
    /// ### Known problems
    /// The lint is only partially aware of the required MSRV for items that were originally in `std` but moved
    /// to `core`.
    ///
    /// ### Example
    /// ```no_run
    /// # extern crate alloc;
    /// use alloc::slice::from_ref;
    /// ```
    /// Use instead:
    /// ```no_run
    /// use core::slice::from_ref;
    /// ```
    #[clippy::version = "1.64.0"]
    pub ALLOC_INSTEAD_OF_CORE,
    restriction,
    "type is imported from alloc when available in core"
}

pub struct StdReexports {
    lint_points: Option<(Span, Vec<LintPoint>)>,
    msrv: Msrv,
}

impl StdReexports {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            lint_points: Option::default(),
            msrv: conf.msrv,
        }
    }

    fn lint_if_finish(&mut self, cx: &LateContext<'_>, krate: Span, lint_point: LintPoint) {
        match &mut self.lint_points {
            Some((prev_krate, prev_lints)) if prev_krate.overlaps(krate) => {
                prev_lints.push(lint_point);
            },
            _ => emit_lints(cx, self.lint_points.replace((krate, vec![lint_point]))),
        }
    }
}

impl_lint_pass!(StdReexports => [STD_INSTEAD_OF_CORE, STD_INSTEAD_OF_ALLOC, ALLOC_INSTEAD_OF_CORE]);

#[derive(Debug)]
enum LintPoint {
    Available(Span, &'static Lint, &'static str, &'static str),
    Conflict,
}

impl<'tcx> LateLintPass<'tcx> for StdReexports {
    fn check_path(&mut self, cx: &LateContext<'tcx>, path: &Path<'tcx>, _: HirId) {
        if let Res::Def(def_kind, def_id) = path.res
            && let Some(first_segment) = get_first_segment(path)
            && is_stable(cx, def_id, self.msrv)
            && !path.span.in_external_macro(cx.sess().source_map())
            && !is_from_proc_macro(cx, &first_segment.ident)
            && !matches!(def_kind, DefKind::Macro(_))
            && let Some(last_segment) = path.segments.last()
            && let Res::Def(DefKind::Mod, crate_def_id) = first_segment.res
            && crate_def_id.is_crate_root()
        {
            let (lint, used_mod, replace_with) = match first_segment.ident.name {
                sym::std => match cx.tcx.crate_name(def_id.krate) {
                    sym::core => (STD_INSTEAD_OF_CORE, "std", "core"),
                    sym::alloc => (STD_INSTEAD_OF_ALLOC, "std", "alloc"),
                    _ => {
                        self.lint_if_finish(cx, first_segment.ident.span, LintPoint::Conflict);
                        return;
                    },
                },
                sym::alloc => {
                    if cx.tcx.crate_name(def_id.krate) == sym::core {
                        (ALLOC_INSTEAD_OF_CORE, "alloc", "core")
                    } else {
                        self.lint_if_finish(cx, first_segment.ident.span, LintPoint::Conflict);
                        return;
                    }
                },
                _ => {
                    self.lint_if_finish(cx, first_segment.ident.span, LintPoint::Conflict);
                    return;
                },
            };

            self.lint_if_finish(
                cx,
                first_segment.ident.span,
                LintPoint::Available(last_segment.ident.span, lint, used_mod, replace_with),
            );
        }
    }

    fn check_block_post(&mut self, cx: &LateContext<'tcx>, _: &Block<'tcx>) {
        emit_lints(cx, self.lint_points.take());
    }

    fn check_body_post(&mut self, cx: &LateContext<'tcx>, _: &Body<'tcx>) {
        emit_lints(cx, self.lint_points.take());
    }

    fn check_crate_post(&mut self, cx: &LateContext<'tcx>) {
        emit_lints(cx, self.lint_points.take());
    }
}

fn emit_lints(cx: &LateContext<'_>, lint_points: Option<(Span, Vec<LintPoint>)>) {
    let Some((krate_span, lint_points)) = lint_points else {
        return;
    };

    let mut lint: Option<(&'static Lint, &'static str, &'static str)> = None;
    let mut has_conflict = false;
    for lint_point in &lint_points {
        match lint_point {
            LintPoint::Available(_, l, used_mod, replace_with)
                if lint.is_none_or(|(prev_l, ..)| l.name == prev_l.name) =>
            {
                lint = Some((l, used_mod, replace_with));
            },
            _ => {
                has_conflict = true;
                break;
            },
        }
    }

    if !has_conflict && let Some((lint, used_mod, replace_with)) = lint {
        span_lint_and_sugg(
            cx,
            lint,
            krate_span,
            format!("used import from `{used_mod}` instead of `{replace_with}`"),
            format!("consider importing the item from `{replace_with}`"),
            (*replace_with).to_string(),
            Applicability::MachineApplicable,
        );
        return;
    }

    for lint_point in lint_points {
        let LintPoint::Available(span, lint, used_mod, replace_with) = lint_point else {
            continue;
        };
        span_lint_and_help(
            cx,
            lint,
            span,
            format!("used import from `{used_mod}` instead of `{replace_with}`"),
            None,
            format!("consider importing the item from `{replace_with}`"),
        );
    }
}

/// Returns the first named segment of a [`Path`].
///
/// If this is a global path (such as `::std::fmt::Debug`), then the segment after [`kw::PathRoot`]
/// is returned.
fn get_first_segment<'tcx>(path: &Path<'tcx>) -> Option<&'tcx PathSegment<'tcx>> {
    match path.segments {
        // A global path will have PathRoot as the first segment. In this case, return the segment after.
        [x, y, ..] if x.ident.name == kw::PathRoot => Some(y),
        [x, ..] => Some(x),
        _ => None,
    }
}

/// Checks if all ancestors of `def_id` meet `msrv` to avoid linting [unstable moves](https://github.com/rust-lang/rust/pull/95956)
/// or now stable moves that were once unstable.
///
/// Does not catch individually moved items
fn is_stable(cx: &LateContext<'_>, mut def_id: DefId, msrv: Msrv) -> bool {
    loop {
        if let Some(stability) = cx.tcx.lookup_stability(def_id)
            && let StabilityLevel::Stable {
                since,
                allowed_through_unstable_modules: None,
            } = stability.level
        {
            let stable = match since {
                StableSince::Version(v) => msrv.meets(cx, v),
                StableSince::Current => msrv.current(cx).is_none(),
                StableSince::Err(_) => false,
            };

            if !stable {
                return false;
            }
        }

        match cx.tcx.opt_parent(def_id) {
            Some(parent) => def_id = parent,
            None => return true,
        }
    }
}
