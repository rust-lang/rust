//! A pass that annotates every item and method with its stability level,
//! propagating default levels lexically from parent to children ast nodes.

pub use self::StabilityLevel::*;

use rustc_ast::CRATE_NODE_ID;
use rustc_attr::{self as attr, ConstStability, Deprecation, Stability};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::{Applicability, DiagnosticBuilder};
use rustc_feature::GateIssue;
use rustc_hir::def_id::CrateNum;
use rustc_hir::{self, HirId};
use rustc_session::lint::builtin::{DEPRECATED, DEPRECATED_IN_FUTURE, SOFT_UNSTABLE};
use rustc_session::lint::{BuiltinLintDiagnostics, Lint, LintBuffer};
use rustc_session::parse::feature_err_issue;
use rustc_session::{DiagnosticMessageId, Session};
use rustc_span::symbol::Symbol;
use rustc_span::{MultiSpan, Span};

use std::num::NonZeroU32;

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum StabilityLevel {
    Unstable,
    Stable,
}

impl StabilityLevel {
    pub fn from_attr_level(level: &attr::StabilityLevel) -> Self {
        if level.is_stable() { Stable } else { Unstable }
    }
}

/// An entry in the `depr_map`.
#[derive(Clone, HashStable_Generic)]
pub struct DeprecationEntry {
    /// The metadata of the attribute associated with this entry.
    pub attr: Deprecation,
    /// The `DefId` where the attr was originally attached. `None` for non-local
    /// `DefId`'s.
    origin: Option<HirId>,
}

impl DeprecationEntry {
    pub fn local(attr: Deprecation, id: HirId) -> DeprecationEntry {
        DeprecationEntry { attr, origin: Some(id) }
    }

    pub fn external(attr: Deprecation) -> DeprecationEntry {
        DeprecationEntry { attr, origin: None }
    }

    pub fn same_origin(&self, other: &DeprecationEntry) -> bool {
        match (self.origin, other.origin) {
            (Some(o1), Some(o2)) => o1 == o2,
            _ => false,
        }
    }
}

/// A stability index, giving the stability level for items and methods.
#[derive(HashStable_Generic)]
pub struct Index<'tcx> {
    /// This is mostly a cache, except the stabilities of local items
    /// are filled by the annotator.
    pub stab_map: FxHashMap<HirId, &'tcx Stability>,
    pub const_stab_map: FxHashMap<HirId, &'tcx ConstStability>,
    pub depr_map: FxHashMap<HirId, DeprecationEntry>,

    /// Maps for each crate whether it is part of the staged API.
    pub staged_api: FxHashMap<CrateNum, bool>,

    /// Features enabled for this crate.
    pub active_features: FxHashSet<Symbol>,
}

impl<'tcx> Index<'tcx> {
    pub fn local_stability(&self, id: HirId) -> Option<&'tcx Stability> {
        self.stab_map.get(&id).cloned()
    }

    pub fn local_const_stability(&self, id: HirId) -> Option<&'tcx ConstStability> {
        self.const_stab_map.get(&id).cloned()
    }

    pub fn local_deprecation_entry(&self, id: HirId) -> Option<DeprecationEntry> {
        self.depr_map.get(&id).cloned()
    }
}

pub fn report_unstable(
    sess: &Session,
    feature: Symbol,
    reason: Option<Symbol>,
    issue: Option<NonZeroU32>,
    is_soft: bool,
    span: Span,
    soft_handler: impl FnOnce(&'static Lint, Span, &str),
) {
    let msg = match reason {
        Some(r) => format!("use of unstable library feature '{}': {}", feature, r),
        None => format!("use of unstable library feature '{}'", &feature),
    };

    let msp: MultiSpan = span.into();
    let sm = &sess.parse_sess.source_map();
    let span_key = msp.primary_span().and_then(|sp: Span| {
        if !sp.is_dummy() {
            let file = sm.lookup_char_pos(sp.lo()).file;
            if file.is_imported() { None } else { Some(span) }
        } else {
            None
        }
    });

    let error_id = (DiagnosticMessageId::StabilityId(issue), span_key, msg.clone());
    let fresh = sess.one_time_diagnostics.borrow_mut().insert(error_id);
    if fresh {
        if is_soft {
            soft_handler(SOFT_UNSTABLE, span, &msg)
        } else {
            feature_err_issue(&sess.parse_sess, feature, span, GateIssue::Library(issue), &msg)
                .emit();
        }
    }
}

/// Checks whether an item marked with `deprecated(since="X")` is currently
/// deprecated (i.e., whether X is not greater than the current rustc version).
pub fn deprecation_in_effect(is_since_rustc_version: bool, since: Option<&str>) -> bool {
    let since = if let Some(since) = since {
        if is_since_rustc_version {
            since
        } else {
            // We assume that the deprecation is in effect if it's not a
            // rustc version.
            return true;
        }
    } else {
        // If since attribute is not set, then we're definitely in effect.
        return true;
    };
    fn parse_version(ver: &str) -> Vec<u32> {
        // We ignore non-integer components of the version (e.g., "nightly").
        ver.split(|c| c == '.' || c == '-').flat_map(|s| s.parse()).collect()
    }

    if let Some(rustc) = option_env!("CFG_RELEASE") {
        let since: Vec<u32> = parse_version(&since);
        let rustc: Vec<u32> = parse_version(rustc);
        // We simply treat invalid `since` attributes as relating to a previous
        // Rust version, thus always displaying the warning.
        if since.len() != 3 {
            return true;
        }
        since <= rustc
    } else {
        // By default, a deprecation warning applies to
        // the current version of the compiler.
        true
    }
}

pub fn deprecation_suggestion(
    diag: &mut DiagnosticBuilder<'_>,
    kind: &str,
    suggestion: Option<Symbol>,
    span: Span,
) {
    if let Some(suggestion) = suggestion {
        diag.span_suggestion(
            span,
            &format!("replace the use of the deprecated {}", kind),
            suggestion.to_string(),
            Applicability::MachineApplicable,
        );
    }
}

pub fn deprecation_message(depr: &Deprecation, kind: &str, path: &str) -> (String, &'static Lint) {
    let (message, lint) = if deprecation_in_effect(
        depr.is_since_rustc_version,
        depr.since.map(Symbol::as_str).as_deref(),
    ) {
        (format!("use of deprecated {} `{}`", kind, path), DEPRECATED)
    } else {
        (
            format!(
                "use of {} `{}` that will be deprecated in future version {}",
                kind,
                path,
                depr.since.unwrap()
            ),
            DEPRECATED_IN_FUTURE,
        )
    };
    let message = match depr.note {
        Some(reason) => format!("{}: {}", message, reason),
        None => message,
    };
    (message, lint)
}

pub fn early_report_deprecation<'a>(
    lint_buffer: &'a mut LintBuffer,
    message: &str,
    suggestion: Option<Symbol>,
    lint: &'static Lint,
    span: Span,
) {
    if span.in_derive_expansion() {
        return;
    }

    let diag = BuiltinLintDiagnostics::DeprecatedMacro(suggestion, span);
    lint_buffer.buffer_lint_with_diagnostic(lint, CRATE_NODE_ID, span, message, diag);
}
