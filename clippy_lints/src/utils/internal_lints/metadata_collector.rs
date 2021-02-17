//! This lint is used to collect metadata about clippy lints. This metadata is exported as a json
//! file and then used to generate the [clippy lint list](https://rust-lang.github.io/rust-clippy/master/index.html)
//!
//! This module and therefor the entire lint is guarded by a feature flag called
//! `metadata-collector-lint`
//!
//! The module transforms all lint names to ascii lowercase to ensure that we don't have mismatches
//! during any comparison or mapping. (Please take care of this, it's not fun to spend time on such
//! a simple mistake)
//!
//! The metadata currently contains:
//! - [x] TODO The lint declaration line for [#1303](https://github.com/rust-lang/rust-clippy/issues/1303)
//!   and [#6492](https://github.com/rust-lang/rust-clippy/issues/6492)
//! - [ ] TODO The Applicability for each lint for [#4310](https://github.com/rust-lang/rust-clippy/issues/4310)

// # Applicability
// - TODO xFrednet 2021-01-17: Find all methods that take and modify applicability or predefine
//   them?
// - TODO xFrednet 2021-01-17: Find lint emit and collect applicability
// # NITs
// - TODO xFrednet 2021-02-13: Collect depreciations and maybe renames

use if_chain::if_chain;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir::{self as hir, ExprKind, Item, ItemKind, Mutability};
use rustc_lint::{CheckLintNameResult, LateContext, LateLintPass, LintContext, LintId};
use rustc_middle::ty::BorrowKind;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::{sym, Loc, Span, Symbol};
use rustc_trait_selection::infer::TyCtxtInferExt;
use rustc_typeck::expr_use_visitor::{ConsumeMode, Delegate, ExprUseVisitor, PlaceWithHirId};
use rustc_typeck::hir_ty_to_ty;
use serde::Serialize;
use std::fs::{self, OpenOptions};
use std::io::prelude::*;
use std::path::Path;

use crate::utils::internal_lints::is_lint_ref_type;
use crate::utils::{
    last_path_segment, match_function_call, match_type, path_to_local_id, paths, span_lint, walk_ptrs_ty_depth,
};

/// This is the output file of the lint collector.
const OUTPUT_FILE: &str = "metadata_collection.json";
/// These lints are excluded from the export.
const BLACK_LISTED_LINTS: [&str; 2] = ["lint_author", "deep_code_inspection"];

// TODO xFrednet 2021-02-15: `span_lint_and_then` & `span_lint_hir_and_then` requires special
// handling
#[rustfmt::skip]
const LINT_EMISSION_FUNCTIONS: [&[&str]; 5] = [
    &["clippy_lints", "utils", "diagnostics", "span_lint"],
    &["clippy_lints", "utils", "diagnostics", "span_lint_and_help"],
    &["clippy_lints", "utils", "diagnostics", "span_lint_and_note"],
    &["clippy_lints", "utils", "diagnostics", "span_lint_hir"],
    &["clippy_lints", "utils", "diagnostics", "span_lint_and_sugg"],
];

declare_clippy_lint! {
    /// **What it does:** Collects metadata about clippy lints for the website.
    ///
    /// This lint will be used to report problems of syntax parsing. You should hopefully never
    /// see this but never say never I guess ^^
    ///
    /// **Why is this bad?** This is not a bad thing but definitely a hacky way to do it. See
    /// issue [#4310](https://github.com/rust-lang/rust-clippy/issues/4310) for a discussion
    /// about the implementation.
    ///
    /// **Known problems:** Hopefully none. It would be pretty uncool to have a problem here :)
    ///
    /// **Example output:**
    /// ```json,ignore
    /// {
    ///     "id": "internal_metadata_collector",
    ///     "id_span": {
    ///         "path": "clippy_lints/src/utils/internal_lints/metadata_collector.rs",
    ///         "line": 1
    ///     },
    ///     "group": "clippy::internal",
    ///     "docs": " **What it does:** Collects metadata about clippy lints for the website. [...] "
    /// }
    /// ```
    pub INTERNAL_METADATA_COLLECTOR,
    internal,
    "A busy bee collection metadata about lints"
}

impl_lint_pass!(MetadataCollector => [INTERNAL_METADATA_COLLECTOR]);

#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone, Default)]
pub struct MetadataCollector {
    lints: Vec<LintMetadata>,
    applicability_into: FxHashMap<String, ApplicabilityInfo>,
}

impl Drop for MetadataCollector {
    /// You might ask: How hacky is this?
    /// My answer:     YES
    fn drop(&mut self) {
        if self.lints.is_empty() {
            return;
        }

        let mut applicability_info = std::mem::take(&mut self.applicability_into);

        // Mapping the final data
        self.lints
            .iter_mut()
            .for_each(|x| x.applicability = applicability_info.remove(&x.id));

        // Outputting
        if Path::new(OUTPUT_FILE).exists() {
            fs::remove_file(OUTPUT_FILE).unwrap();
        }
        let mut file = OpenOptions::new().write(true).create(true).open(OUTPUT_FILE).unwrap();
        writeln!(file, "{}", serde_json::to_string_pretty(&self.lints).unwrap()).unwrap();
    }
}

#[derive(Debug, Clone, Serialize)]
struct LintMetadata {
    id: String,
    id_span: SerializableSpan,
    group: String,
    docs: String,
    /// This field is only used in the output and will only be
    /// mapped shortly before the actual output.
    applicability: Option<ApplicabilityInfo>,
}

impl LintMetadata {
    fn new(id: String, id_span: SerializableSpan, group: String, docs: String) -> Self {
        Self {
            id,
            id_span,
            group,
            docs,
            applicability: None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct SerializableSpan {
    path: String,
    line: usize,
}

impl SerializableSpan {
    fn from_item(cx: &LateContext<'_>, item: &Item<'_>) -> Self {
        Self::from_span(cx, item.ident.span)
    }

    fn from_span(cx: &LateContext<'_>, span: Span) -> Self {
        let loc: Loc = cx.sess().source_map().lookup_char_pos(span.lo());

        Self {
            path: format!("{}", loc.file.name),
            line: loc.line,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize)]
struct ApplicabilityInfo {
    /// Indicates if any of the lint emissions uses multiple spans. This is related to
    /// [rustfix#141](https://github.com/rust-lang/rustfix/issues/141) as such suggestions can
    /// currently not be applied automatically.
    has_multi_suggestion: bool,
    /// These are all the available applicability values for the lint suggestions
    applicabilities: FxHashSet<String>,
}

fn log_to_file(msg: &str) {
    let mut file = OpenOptions::new()
        .write(true)
        .append(true)
        .create(true)
        .open("metadata-lint.log")
        .unwrap();

    write!(file, "{}", msg).unwrap();
}

impl<'tcx> LateLintPass<'tcx> for MetadataCollector {
    /// Collecting lint declarations like:
    /// ```rust, ignore
    /// declare_clippy_lint! {
    ///     /// **What it does:** Something IDK.
    ///     pub SOME_LINT,
    ///     internal,
    ///     "Who am I?"
    /// }
    /// ```
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        if_chain! {
            if let ItemKind::Static(ref ty, Mutability::Not, body_id) = item.kind;
            if is_lint_ref_type(cx, ty);
            let expr = &cx.tcx.hir().body(body_id).value;
            if let ExprKind::AddrOf(_, _, ref inner_exp) = expr.kind;
            if let ExprKind::Struct(_, _, _) = inner_exp.kind;
            then {
                let lint_name = sym_to_string(item.ident.name).to_ascii_lowercase();
                if BLACK_LISTED_LINTS.contains(&lint_name.as_str()) {
                    return;
                }

                let group: String;
                let result = cx.lint_store.check_lint_name(lint_name.as_str(), Some(sym::clippy));
                if let CheckLintNameResult::Tool(Ok(lint_lst)) = result {
                    if let Some(group_some) = get_lint_group(cx, lint_lst[0]) {
                        group = group_some;
                    } else {
                        lint_collection_error_item(cx, item, "Unable to determine lint group");
                        return;
                    }
                } else {
                    lint_collection_error_item(cx, item, "Unable to find lint in lint_store");
                    return;
                }

                let docs: String;
                if let Some(docs_some) = extract_attr_docs(item) {
                    docs = docs_some;
                } else {
                    lint_collection_error_item(cx, item, "could not collect the lint documentation");
                    return;
                };

                self.lints.push(LintMetadata::new(
                    lint_name,
                    SerializableSpan::from_item(cx, item),
                    group,
                    docs,
                ));
            }
        }
    }

    /// Collecting constant applicability from the actual lint emissions
    ///
    /// Example:
    /// ```rust, ignore
    /// span_lint_and_sugg(
    ///     cx,
    ///     SOME_LINT,
    ///     item.span,
    ///     "Le lint message",
    ///     "Here comes help:",
    ///     "#![allow(clippy::all)]",
    ///     Applicability::MachineApplicable, // <-- Extracts this constant value
    /// );
    /// ```
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        if let Some(args) = match_simple_lint_emission(cx, expr) {
            if let Some((lint_name, mut applicability)) = extract_emission_info(cx, args) {
                let app_info = self.applicability_into.entry(lint_name).or_default();
                applicability.drain(..).for_each(|x| {
                    app_info.applicabilities.insert(x);
                });
            } else {
                lint_collection_error_span(cx, expr.span, "I found this but I can't get the lint or applicability");
            }
        }
    }

    /// Tracking and hopefully collecting dynamic applicability values
    ///
    /// Example:
    /// ```rust, ignore
    /// // vvv Applicability value to track
    /// let mut applicability = Applicability::MachineApplicable;
    /// // vvv Value Mutation
    /// let suggestion = snippet_with_applicability(cx, expr.span, "_", &mut applicability);
    /// // vvv Emission to link the value to the lint
    /// span_lint_and_sugg(
    ///     cx,
    ///     SOME_LINT,
    ///     expr.span,
    ///     "This can be improved",
    ///     "try",
    ///     suggestion,
    ///     applicability,
    /// );
    /// ```
    fn check_local(&mut self, cx: &LateContext<'tcx>, local: &'tcx hir::Local<'tcx>) {
        if let Some(tc) = cx.maybe_typeck_results() {
            // TODO xFrednet 2021-02-14: support nested applicability (only in tuples)
            let local_ty = if let Some(ty) = local.ty {
                hir_ty_to_ty(cx.tcx, ty)
            } else if let Some(init) = local.init {
                tc.expr_ty(init)
            } else {
                return;
            };

            if_chain! {
                if match_type(cx, local_ty, &paths::APPLICABILITY);
                if let Some(body) = get_parent_body(cx, local.hir_id);
                then {
                    let span = SerializableSpan::from_span(cx, local.span);
                    let local_str = crate::utils::snippet(cx, local.span, "_");
                    let value_life = format!("{} -- {}:{}\n", local_str, span.path.rsplit('/').next().unwrap_or_default(), span.line);
                    let value_hir_id = local.pat.hir_id;
                    let mut tracker = ValueTracker {cx, value_hir_id, value_life};

                    cx.tcx.infer_ctxt().enter(|infcx| {
                        let body_owner_id = cx.tcx.hir().body_owner_def_id(body.id());
                        ExprUseVisitor::new(
                            &mut tracker,
                            &infcx,
                            body_owner_id,
                            cx.param_env,
                            cx.typeck_results()
                        )
                        .consume_body(body);
                    });

                    log_to_file(&tracker.value_life);
                    lint_collection_error_span(cx, local.span, "Applicability value found");
                }
            }
        }
    }
}

fn get_parent_body<'a, 'tcx>(cx: &'a LateContext<'tcx>, id: hir::HirId) -> Option<&'tcx hir::Body<'tcx>> {
    let map = cx.tcx.hir();

    map.parent_iter(id)
        .find_map(|(parent, _)| map.maybe_body_owned_by(parent))
        .map(|body| map.body(body))
}

fn sym_to_string(sym: Symbol) -> String {
    sym.as_str().to_string()
}

/// This function collects all documentation that has been added to an item using
/// `#[doc = r""]` attributes. Several attributes are aggravated using line breaks
///
/// ```ignore
/// #[doc = r"Hello world!"]
/// #[doc = r"=^.^="]
/// struct SomeItem {}
/// ```
///
/// Would result in `Hello world!\n=^.^=\n`
fn extract_attr_docs(item: &Item<'_>) -> Option<String> {
    item.attrs
        .iter()
        .filter_map(|ref x| x.doc_str().map(|sym| sym.as_str().to_string()))
        .reduce(|mut acc, sym| {
            acc.push_str(&sym);
            acc.push('\n');
            acc
        })
}

fn get_lint_group(cx: &LateContext<'_>, lint_id: LintId) -> Option<String> {
    for (group_name, lints, _) in &cx.lint_store.get_lint_groups() {
        if lints.iter().any(|x| *x == lint_id) {
            return Some((*group_name).to_string());
        }
    }

    None
}

// ==================================================================
// Lint emission
// ==================================================================
fn lint_collection_error_item(cx: &LateContext<'_>, item: &Item<'_>, message: &str) {
    span_lint(
        cx,
        INTERNAL_METADATA_COLLECTOR,
        item.ident.span,
        &format!("Metadata collection error for `{}`: {}", item.ident.name, message),
    );
}

fn lint_collection_error_span(cx: &LateContext<'_>, span: Span, message: &str) {
    span_lint(
        cx,
        INTERNAL_METADATA_COLLECTOR,
        span,
        &format!("Metadata collection error: {}", message),
    );
}

// ==================================================================
// Applicability
// ==================================================================
fn match_simple_lint_emission<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'_>,
) -> Option<&'tcx [hir::Expr<'tcx>]> {
    LINT_EMISSION_FUNCTIONS
        .iter()
        .find_map(|emission_fn| match_function_call(cx, expr, emission_fn))
}

/// This returns the lint name and the possible applicability of this emission
fn extract_emission_info<'tcx>(cx: &LateContext<'tcx>, args: &[hir::Expr<'_>]) -> Option<(String, Vec<String>)> {
    let mut lint_name = None;
    let mut applicability = None;

    for arg in args {
        let (arg_ty, _) = walk_ptrs_ty_depth(cx.typeck_results().expr_ty(&arg));

        if match_type(cx, arg_ty, &paths::LINT) {
            // If we found the lint arg, extract the lint name
            if let ExprKind::Path(ref lint_path) = arg.kind {
                lint_name = Some(last_path_segment(lint_path).ident.name)
            }
        } else if match_type(cx, arg_ty, &paths::APPLICABILITY) {
            if let ExprKind::Path(ref path) = arg.kind {
                applicability = Some(last_path_segment(path).ident.name)
            }
        }
    }

    lint_name.map(|lint_name| {
        (
            sym_to_string(lint_name).to_ascii_lowercase(),
            applicability.map(sym_to_string).map_or_else(Vec::new, |x| vec![x]),
        )
    })
}

struct ValueTracker<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    value_hir_id: hir::HirId,
    value_life: String,
}

impl<'a, 'tcx> ValueTracker<'a, 'tcx> {
    fn is_value_expr(&self, expr_id: hir::HirId) -> bool {
        match self.cx.tcx.hir().find(expr_id) {
            Some(hir::Node::Expr(expr)) => path_to_local_id(expr, self.value_hir_id),
            _ => false,
        }
    }
}

impl<'a, 'tcx> Delegate<'tcx> for ValueTracker<'a, 'tcx> {
    fn consume(&mut self, _place_with_id: &PlaceWithHirId<'tcx>, expr_id: hir::HirId, _: ConsumeMode) {
        if self.is_value_expr(expr_id) {
            // TODO xFrednet 2021-02-17: Check if lint emission and extract lint ID
            todo!();
        }
    }

    fn borrow(&mut self, _place_with_id: &PlaceWithHirId<'tcx>, expr_id: hir::HirId, bk: BorrowKind) {
        if self.is_value_expr(expr_id) {
            if let BorrowKind::MutBorrow = bk {
                // TODO xFrednet 2021-02-17: Save the function
                todo!();
            }
        }
    }

    fn mutate(&mut self, _assignee_place: &PlaceWithHirId<'tcx>, expr_id: hir::HirId) {
        if self.is_value_expr(expr_id) {
            // TODO xFrednet 2021-02-17: Save the new value as a mutation
            todo!();
        }
    }
}
