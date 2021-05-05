//! This lint is used to collect metadata about clippy lints. This metadata is exported as a json
//! file and then used to generate the [clippy lint list](https://rust-lang.github.io/rust-clippy/master/index.html)
//!
//! This module and therefor the entire lint is guarded by a feature flag called
//! `metadata-collector-lint`
//!
//! The module transforms all lint names to ascii lowercase to ensure that we don't have mismatches
//! during any comparison or mapping. (Please take care of this, it's not fun to spend time on such
//! a simple mistake)

// # NITs
// - TODO xFrednet 2021-02-13: Collect depreciations and maybe renames

use if_chain::if_chain;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::{
    self as hir, def::DefKind, intravisit, intravisit::Visitor, ExprKind, Item, ItemKind, Mutability, QPath,
};
use rustc_lint::{CheckLintNameResult, LateContext, LateLintPass, LintContext, LintId};
use rustc_middle::hir::map::Map;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::{sym, Loc, Span, Symbol};
use serde::{ser::SerializeStruct, Serialize, Serializer};
use std::collections::BinaryHeap;
use std::fs::{self, OpenOptions};
use std::io::prelude::*;
use std::path::Path;

use crate::utils::internal_lints::is_lint_ref_type;
use clippy_utils::{
    diagnostics::span_lint, last_path_segment, match_function_call, match_path, paths, ty::match_type,
    ty::walk_ptrs_ty_depth,
};

/// This is the output file of the lint collector.
const OUTPUT_FILE: &str = "../util/gh-pages/metadata_collection.json";
/// These lints are excluded from the export.
const BLACK_LISTED_LINTS: [&str; 3] = ["lint_author", "deep_code_inspection", "internal_metadata_collector"];
/// These groups will be ignored by the lint group matcher. This is useful for collections like
/// `clippy::all`
const IGNORED_LINT_GROUPS: [&str; 1] = ["clippy::all"];
/// Lints within this group will be excluded from the collection
const EXCLUDED_LINT_GROUPS: [&str; 1] = ["clippy::internal"];

const LINT_EMISSION_FUNCTIONS: [&[&str]; 7] = [
    &["clippy_utils", "diagnostics", "span_lint"],
    &["clippy_utils", "diagnostics", "span_lint_and_help"],
    &["clippy_utils", "diagnostics", "span_lint_and_note"],
    &["clippy_utils", "diagnostics", "span_lint_hir"],
    &["clippy_utils", "diagnostics", "span_lint_and_sugg"],
    &["clippy_utils", "diagnostics", "span_lint_and_then"],
    &["clippy_utils", "diagnostics", "span_lint_hir_and_then"],
];
const SUGGESTION_DIAGNOSTIC_BUILDER_METHODS: [(&str, bool); 9] = [
    ("span_suggestion", false),
    ("span_suggestion_short", false),
    ("span_suggestion_verbose", false),
    ("span_suggestion_hidden", false),
    ("tool_only_span_suggestion", false),
    ("multipart_suggestion", true),
    ("multipart_suggestions", true),
    ("tool_only_multipart_suggestion", true),
    ("span_suggestions", true),
];
const SUGGESTION_FUNCTIONS: [&[&str]; 2] = [
    &["clippy_utils", "diagnostics", "multispan_sugg"],
    &["clippy_utils", "diagnostics", "multispan_sugg_with_applicability"],
];

/// The index of the applicability name of `paths::APPLICABILITY_VALUES`
const APPLICABILITY_NAME_INDEX: usize = 2;

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
    internal_warn,
    "A busy bee collection metadata about lints"
}

impl_lint_pass!(MetadataCollector => [INTERNAL_METADATA_COLLECTOR]);

#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone, Default)]
pub struct MetadataCollector {
    /// All collected lints
    ///
    /// We use a Heap here to have the lints added in alphabetic order in the export
    lints: BinaryHeap<LintMetadata>,
    applicability_info: FxHashMap<String, ApplicabilityInfo>,
}

impl Drop for MetadataCollector {
    /// You might ask: How hacky is this?
    /// My answer:     YES
    fn drop(&mut self) {
        // The metadata collector gets dropped twice, this makes sure that we only write
        // when the list is full
        if self.lints.is_empty() {
            return;
        }

        let mut applicability_info = std::mem::take(&mut self.applicability_info);

        // Mapping the final data
        let mut lints = std::mem::take(&mut self.lints).into_sorted_vec();
        lints
            .iter_mut()
            .for_each(|x| x.applicability = applicability_info.remove(&x.id));

        // Outputting
        if Path::new(OUTPUT_FILE).exists() {
            fs::remove_file(OUTPUT_FILE).unwrap();
        }
        let mut file = OpenOptions::new().write(true).create(true).open(OUTPUT_FILE).unwrap();
        writeln!(file, "{}", serde_json::to_string_pretty(&lints).unwrap()).unwrap();
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq, PartialOrd, Ord)]
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

#[derive(Debug, Clone, Serialize, PartialEq, Eq, PartialOrd, Ord)]
struct SerializableSpan {
    path: String,
    line: usize,
}

impl std::fmt::Display for SerializableSpan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.path.rsplit('/').next().unwrap_or_default(), self.line)
    }
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

#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
struct ApplicabilityInfo {
    /// Indicates if any of the lint emissions uses multiple spans. This is related to
    /// [rustfix#141](https://github.com/rust-lang/rustfix/issues/141) as such suggestions can
    /// currently not be applied automatically.
    is_multi_part_suggestion: bool,
    applicability: Option<usize>,
}

impl Serialize for ApplicabilityInfo {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let index = self.applicability.unwrap_or_default();

        let mut s = serializer.serialize_struct("ApplicabilityInfo", 2)?;
        s.serialize_field("is_multi_part_suggestion", &self.is_multi_part_suggestion)?;
        s.serialize_field(
            "applicability",
            &paths::APPLICABILITY_VALUES[index][APPLICABILITY_NAME_INDEX],
        )?;
        s.end()
    }
}

impl<'hir> LateLintPass<'hir> for MetadataCollector {
    /// Collecting lint declarations like:
    /// ```rust, ignore
    /// declare_clippy_lint! {
    ///     /// **What it does:** Something IDK.
    ///     pub SOME_LINT,
    ///     internal,
    ///     "Who am I?"
    /// }
    /// ```
    fn check_item(&mut self, cx: &LateContext<'hir>, item: &'hir Item<'_>) {
        if_chain! {
            // item validation
            if let ItemKind::Static(ref ty, Mutability::Not, _) = item.kind;
            if is_lint_ref_type(cx, ty);
            // blacklist check
            let lint_name = sym_to_string(item.ident.name).to_ascii_lowercase();
            if !BLACK_LISTED_LINTS.contains(&lint_name.as_str());
            // metadata extraction
            if let Some(group) = get_lint_group_or_lint(cx, &lint_name, item);
            if let Some(docs) = extract_attr_docs_or_lint(cx, item);
            then {
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
    fn check_expr(&mut self, cx: &LateContext<'hir>, expr: &'hir hir::Expr<'_>) {
        if let Some(args) = match_lint_emission(cx, expr) {
            let mut emission_info = extract_emission_info(cx, args);
            if emission_info.is_empty() {
                // See:
                // - src/misc.rs:734:9
                // - src/methods/mod.rs:3545:13
                // - src/methods/mod.rs:3496:13
                // We are basically unable to resolve the lint name it self.
                return;
            }

            for (lint_name, applicability, is_multi_part) in emission_info.drain(..) {
                let app_info = self.applicability_info.entry(lint_name).or_default();
                app_info.applicability = applicability;
                app_info.is_multi_part_suggestion = is_multi_part;
            }
        }
    }
}

// ==================================================================
// Lint definition extraction
// ==================================================================
fn sym_to_string(sym: Symbol) -> String {
    sym.as_str().to_string()
}

fn extract_attr_docs_or_lint(cx: &LateContext<'_>, item: &Item<'_>) -> Option<String> {
    extract_attr_docs(cx, item).or_else(|| {
        lint_collection_error_item(cx, item, "could not collect the lint documentation");
        None
    })
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
fn extract_attr_docs(cx: &LateContext<'_>, item: &Item<'_>) -> Option<String> {
    cx.tcx
        .hir()
        .attrs(item.hir_id())
        .iter()
        .filter_map(|ref x| x.doc_str().map(|sym| sym.as_str().to_string()))
        .reduce(|mut acc, sym| {
            acc.push_str(&sym);
            acc.push('\n');
            acc
        })
}

fn get_lint_group_or_lint(cx: &LateContext<'_>, lint_name: &str, item: &'hir Item<'_>) -> Option<String> {
    let result = cx.lint_store.check_lint_name(lint_name, Some(sym::clippy));
    if let CheckLintNameResult::Tool(Ok(lint_lst)) = result {
        get_lint_group(cx, lint_lst[0])
            .or_else(|| {
                lint_collection_error_item(cx, item, "Unable to determine lint group");
                None
            })
            .filter(|group| !EXCLUDED_LINT_GROUPS.contains(&group.as_str()))
    } else {
        lint_collection_error_item(cx, item, "Unable to find lint in lint_store");
        None
    }
}

fn get_lint_group(cx: &LateContext<'_>, lint_id: LintId) -> Option<String> {
    for (group_name, lints, _) in &cx.lint_store.get_lint_groups() {
        if IGNORED_LINT_GROUPS.contains(group_name) {
            continue;
        }

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
        &format!("metadata collection error for `{}`: {}", item.ident.name, message),
    );
}

// ==================================================================
// Applicability
// ==================================================================
/// This function checks if a given expression is equal to a simple lint emission function call.
/// It will return the function arguments if the emission matched any function.
fn match_lint_emission<'hir>(cx: &LateContext<'hir>, expr: &'hir hir::Expr<'_>) -> Option<&'hir [hir::Expr<'hir>]> {
    LINT_EMISSION_FUNCTIONS
        .iter()
        .find_map(|emission_fn| match_function_call(cx, expr, emission_fn))
}

fn take_higher_applicability(a: Option<usize>, b: Option<usize>) -> Option<usize> {
    a.map_or(b, |a| a.max(b.unwrap_or_default()).into())
}

fn extract_emission_info<'hir>(
    cx: &LateContext<'hir>,
    args: &'hir [hir::Expr<'hir>],
) -> Vec<(String, Option<usize>, bool)> {
    let mut lints = Vec::new();
    let mut applicability = None;
    let mut multi_part = false;

    for arg in args {
        let (arg_ty, _) = walk_ptrs_ty_depth(cx.typeck_results().expr_ty(&arg));

        if match_type(cx, arg_ty, &paths::LINT) {
            // If we found the lint arg, extract the lint name
            let mut resolved_lints = resolve_lints(cx, arg);
            lints.append(&mut resolved_lints);
        } else if match_type(cx, arg_ty, &paths::APPLICABILITY) {
            applicability = resolve_applicability(cx, arg);
        } else if arg_ty.is_closure() {
            multi_part |= check_is_multi_part(cx, arg);
            // TODO xFrednet 2021-03-01: don't use or_else but rather a comparison
            applicability = applicability.or_else(|| resolve_applicability(cx, arg));
        }
    }

    lints
        .drain(..)
        .map(|lint_name| (lint_name, applicability, multi_part))
        .collect()
}

/// Resolves the possible lints that this expression could reference
fn resolve_lints(cx: &LateContext<'hir>, expr: &'hir hir::Expr<'hir>) -> Vec<String> {
    let mut resolver = LintResolver::new(cx);
    resolver.visit_expr(expr);
    resolver.lints
}

/// This function tries to resolve the linked applicability to the given expression.
fn resolve_applicability(cx: &LateContext<'hir>, expr: &'hir hir::Expr<'hir>) -> Option<usize> {
    let mut resolver = ApplicabilityResolver::new(cx);
    resolver.visit_expr(expr);
    resolver.complete()
}

fn check_is_multi_part(cx: &LateContext<'hir>, closure_expr: &'hir hir::Expr<'hir>) -> bool {
    if let ExprKind::Closure(_, _, body_id, _, _) = closure_expr.kind {
        let mut scanner = IsMultiSpanScanner::new(cx);
        intravisit::walk_body(&mut scanner, cx.tcx.hir().body(body_id));
        return scanner.is_multi_part();
    } else if let Some(local) = get_parent_local(cx, closure_expr) {
        if let Some(local_init) = local.init {
            return check_is_multi_part(cx, local_init);
        }
    }

    false
}

struct LintResolver<'a, 'hir> {
    cx: &'a LateContext<'hir>,
    lints: Vec<String>,
}

impl<'a, 'hir> LintResolver<'a, 'hir> {
    fn new(cx: &'a LateContext<'hir>) -> Self {
        Self {
            cx,
            lints: Vec::<String>::default(),
        }
    }
}

impl<'a, 'hir> intravisit::Visitor<'hir> for LintResolver<'a, 'hir> {
    type Map = Map<'hir>;

    fn nested_visit_map(&mut self) -> intravisit::NestedVisitorMap<Self::Map> {
        intravisit::NestedVisitorMap::All(self.cx.tcx.hir())
    }

    fn visit_expr(&mut self, expr: &'hir hir::Expr<'hir>) {
        if_chain! {
            if let ExprKind::Path(qpath) = &expr.kind;
            if let QPath::Resolved(_, path) = qpath;

            let (expr_ty, _) = walk_ptrs_ty_depth(self.cx.typeck_results().expr_ty(&expr));
            if match_type(self.cx, expr_ty, &paths::LINT);
            then {
                if let hir::def::Res::Def(DefKind::Static, _) = path.res {
                    let lint_name = last_path_segment(qpath).ident.name;
                    self.lints.push(sym_to_string(lint_name).to_ascii_lowercase());
                } else if let Some(local) = get_parent_local(self.cx, expr) {
                    if let Some(local_init) = local.init {
                        intravisit::walk_expr(self, local_init);
                    }
                }
            }
        }

        intravisit::walk_expr(self, expr);
    }
}

/// This visitor finds the highest applicability value in the visited expressions
struct ApplicabilityResolver<'a, 'hir> {
    cx: &'a LateContext<'hir>,
    /// This is the index of hightest `Applicability` for `paths::APPLICABILITY_VALUES`
    applicability_index: Option<usize>,
}

impl<'a, 'hir> ApplicabilityResolver<'a, 'hir> {
    fn new(cx: &'a LateContext<'hir>) -> Self {
        Self {
            cx,
            applicability_index: None,
        }
    }

    fn add_new_index(&mut self, new_index: usize) {
        self.applicability_index = take_higher_applicability(self.applicability_index, Some(new_index));
    }

    fn complete(self) -> Option<usize> {
        self.applicability_index
    }
}

impl<'a, 'hir> intravisit::Visitor<'hir> for ApplicabilityResolver<'a, 'hir> {
    type Map = Map<'hir>;

    fn nested_visit_map(&mut self) -> intravisit::NestedVisitorMap<Self::Map> {
        intravisit::NestedVisitorMap::All(self.cx.tcx.hir())
    }

    fn visit_path(&mut self, path: &'hir hir::Path<'hir>, _id: hir::HirId) {
        for (index, enum_value) in paths::APPLICABILITY_VALUES.iter().enumerate() {
            if match_path(path, enum_value) {
                self.add_new_index(index);
                return;
            }
        }
    }

    fn visit_expr(&mut self, expr: &'hir hir::Expr<'hir>) {
        let (expr_ty, _) = walk_ptrs_ty_depth(self.cx.typeck_results().expr_ty(&expr));

        if_chain! {
            if match_type(self.cx, expr_ty, &paths::APPLICABILITY);
            if let Some(local) = get_parent_local(self.cx, expr);
            if let Some(local_init) = local.init;
            then {
                intravisit::walk_expr(self, local_init);
            }
        };

        // TODO xFrednet 2021-03-01: support function arguments?

        intravisit::walk_expr(self, expr);
    }
}

/// This returns the parent local node if the expression is a reference one
fn get_parent_local(cx: &LateContext<'hir>, expr: &'hir hir::Expr<'hir>) -> Option<&'hir hir::Local<'hir>> {
    if let ExprKind::Path(QPath::Resolved(_, path)) = expr.kind {
        if let hir::def::Res::Local(local_hir) = path.res {
            return get_parent_local_hir_id(cx, local_hir);
        }
    }

    None
}

fn get_parent_local_hir_id(cx: &LateContext<'hir>, hir_id: hir::HirId) -> Option<&'hir hir::Local<'hir>> {
    let map = cx.tcx.hir();

    match map.find(map.get_parent_node(hir_id)) {
        Some(hir::Node::Local(local)) => Some(local),
        Some(hir::Node::Pat(pattern)) => get_parent_local_hir_id(cx, pattern.hir_id),
        _ => None,
    }
}

/// This visitor finds the highest applicability value in the visited expressions
struct IsMultiSpanScanner<'a, 'hir> {
    cx: &'a LateContext<'hir>,
    suggestion_count: usize,
}

impl<'a, 'hir> IsMultiSpanScanner<'a, 'hir> {
    fn new(cx: &'a LateContext<'hir>) -> Self {
        Self {
            cx,
            suggestion_count: 0,
        }
    }

    /// Add a new single expression suggestion to the counter
    fn add_single_span_suggestion(&mut self) {
        self.suggestion_count += 1;
    }

    /// Signals that a suggestion with possible multiple spans was found
    fn add_multi_part_suggestion(&mut self) {
        self.suggestion_count += 2;
    }

    /// Checks if the suggestions include multiple spanns
    fn is_multi_part(&self) -> bool {
        self.suggestion_count > 1
    }
}

impl<'a, 'hir> intravisit::Visitor<'hir> for IsMultiSpanScanner<'a, 'hir> {
    type Map = Map<'hir>;

    fn nested_visit_map(&mut self) -> intravisit::NestedVisitorMap<Self::Map> {
        intravisit::NestedVisitorMap::All(self.cx.tcx.hir())
    }

    fn visit_expr(&mut self, expr: &'hir hir::Expr<'hir>) {
        // Early return if the lint is already multi span
        if self.is_multi_part() {
            return;
        }

        match &expr.kind {
            ExprKind::Call(fn_expr, _args) => {
                let found_function = SUGGESTION_FUNCTIONS
                    .iter()
                    .any(|func_path| match_function_call(self.cx, fn_expr, func_path).is_some());
                if found_function {
                    // These functions are all multi part suggestions
                    self.add_single_span_suggestion()
                }
            },
            ExprKind::MethodCall(path, _path_span, arg, _arg_span) => {
                let (self_ty, _) = walk_ptrs_ty_depth(self.cx.typeck_results().expr_ty(&arg[0]));
                if match_type(self.cx, self_ty, &paths::DIAGNOSTIC_BUILDER) {
                    let called_method = path.ident.name.as_str().to_string();
                    for (method_name, is_multi_part) in &SUGGESTION_DIAGNOSTIC_BUILDER_METHODS {
                        if *method_name == called_method {
                            if *is_multi_part {
                                self.add_multi_part_suggestion();
                            } else {
                                self.add_single_span_suggestion();
                            }
                            break;
                        }
                    }
                }
            },
            _ => {},
        }

        intravisit::walk_expr(self, expr);
    }
}
