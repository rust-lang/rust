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
// - TODO xFrednet 2021-01-17: Find lint emit and collect applicability
//   - TODO xFrednet 2021-02-28:  1x reference to closure
//     - See clippy_lints/src/needless_pass_by_value.rs@NeedlessPassByValue::check_fn
//   - TODO xFrednet 2021-02-28:  4x weird emission forwarding
//     - See clippy_lints/src/enum_variants.rs@EnumVariantNames::check_name
//   - TODO xFrednet 2021-02-28:  6x emission forwarding with local that is initializes from
//     function.
//     - See clippy_lints/src/methods/mod.rs@lint_binary_expr_with_method_call
//   - TODO xFrednet 2021-02-28:  2x lint from local from function call
//     - See clippy_lints/src/misc.rs@check_binary
//   - TODO xFrednet 2021-02-28:  2x lint from local from method call
//     - See clippy_lints/src/non_copy_const.rs@lint
//   - TODO xFrednet 2021-02-28: 20x lint from local
//     - See clippy_lints/src/map_unit_fn.rs@lint_map_unit_fn
// # NITs
// - TODO xFrednet 2021-02-13: Collect depreciations and maybe renames

use if_chain::if_chain;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::{self as hir, intravisit, ExprKind, Item, ItemKind, Mutability, QPath};
use rustc_lint::{CheckLintNameResult, LateContext, LateLintPass, LintContext, LintId};
use rustc_middle::hir::map::Map;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::{sym, Loc, Span, Symbol};
use serde::Serialize;
use std::fs::{self, OpenOptions};
use std::io::prelude::*;
use std::path::Path;

use crate::utils::internal_lints::is_lint_ref_type;
use crate::utils::{
    last_path_segment, match_function_call, match_path, match_type, paths, span_lint, walk_ptrs_ty_depth,
};

/// This is the output file of the lint collector.
const OUTPUT_FILE: &str = "metadata_collection.json";
/// These lints are excluded from the export.
const BLACK_LISTED_LINTS: [&str; 2] = ["lint_author", "deep_code_inspection"];
/// These groups will be ignored by the lint group matcher
const BLACK_LISTED_LINT_GROUP: [&str; 1] = ["clippy::all", "clippy::internal"];

// TODO xFrednet 2021-02-15: `span_lint_and_then` & `span_lint_hir_and_then` requires special
// handling
const SIMPLE_LINT_EMISSION_FUNCTIONS: [&[&str]; 5] = [
    &["clippy_utils", "diagnostics", "span_lint"],
    &["clippy_utils", "diagnostics", "span_lint_and_help"],
    &["clippy_utils", "diagnostics", "span_lint_and_note"],
    &["clippy_utils", "diagnostics", "span_lint_hir"],
    &["clippy_utils", "diagnostics", "span_lint_and_sugg"],
];
const COMPLEX_LINT_EMISSION_FUNCTIONS: [&[&str]; 2] = [
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
    &["clippy_utils", "diagnostics", "mutispan_sugg"],
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

#[derive(Debug, Clone, Default, Serialize)]
struct ApplicabilityInfo {
    /// Indicates if any of the lint emissions uses multiple spans. This is related to
    /// [rustfix#141](https://github.com/rust-lang/rustfix/issues/141) as such suggestions can
    /// currently not be applied automatically.
    is_multi_suggestion: bool,
    applicability: Option<String>,
}

#[allow(dead_code)]
fn log_to_file(msg: &str) {
    let mut file = OpenOptions::new()
        .write(true)
        .append(true)
        .create(true)
        .open("metadata-lint.log")
        .unwrap();

    write!(file, "{}", msg).unwrap();
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
            if let ItemKind::Static(ref ty, Mutability::Not, body_id) = item.kind;
            if is_lint_ref_type(cx, ty);
            let expr = &cx.tcx.hir().body(body_id).value;
            if let ExprKind::AddrOf(_, _, ref inner_exp) = expr.kind;
            if let ExprKind::Struct(_, _, _) = inner_exp.kind;
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
        if let Some(args) = match_simple_lint_emission(cx, expr) {
            if let Some((lint_name, applicability)) = extract_emission_info(cx, args) {
                let app_info = self.applicability_into.entry(lint_name).or_default();
                app_info.applicability = applicability;
            } else {
                lint_collection_error_span(cx, expr.span, "I found this but I can't get the lint or applicability");
            }
        } else if let Some(args) = match_complex_lint_emission(cx, expr) {
            if let Some((lint_name, applicability, is_multi_span)) = extract_complex_emission_info(cx, args) {
                let app_info = self.applicability_into.entry(lint_name).or_default();
                app_info.applicability = applicability;
                app_info.is_multi_suggestion = is_multi_span;
            } else {
                lint_collection_error_span(cx, expr.span, "Look, here ... I have no clue what todo with it");
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
    extract_attr_docs(item).or_else(|| {
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

fn get_lint_group_or_lint(cx: &LateContext<'_>, lint_name: &str, item: &'hir Item<'_>) -> Option<String> {
    let result = cx.lint_store.check_lint_name(lint_name, Some(sym::clippy));
    if let CheckLintNameResult::Tool(Ok(lint_lst)) = result {
        get_lint_group(cx, lint_lst[0]).or_else(|| {
            lint_collection_error_item(cx, item, "Unable to determine lint group");
            None
        })
    } else {
        lint_collection_error_item(cx, item, "Unable to find lint in lint_store");
        None
    }
}

fn get_lint_group(cx: &LateContext<'_>, lint_id: LintId) -> Option<String> {
    for (group_name, lints, _) in &cx.lint_store.get_lint_groups() {
        if BLACK_LISTED_LINT_GROUP.contains(group_name) {
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
/// This function checks if a given expression is equal to a simple lint emission function call.
/// It will return the function arguments if the emission matched any function.
fn match_simple_lint_emission<'hir>(
    cx: &LateContext<'hir>,
    expr: &'hir hir::Expr<'_>,
) -> Option<&'hir [hir::Expr<'hir>]> {
    SIMPLE_LINT_EMISSION_FUNCTIONS
        .iter()
        .find_map(|emission_fn| match_function_call(cx, expr, emission_fn))
}

fn match_complex_lint_emission<'hir>(
    cx: &LateContext<'hir>,
    expr: &'hir hir::Expr<'_>,
) -> Option<&'hir [hir::Expr<'hir>]> {
    COMPLEX_LINT_EMISSION_FUNCTIONS
        .iter()
        .find_map(|emission_fn| match_function_call(cx, expr, emission_fn))
}

/// This returns the lint name and the possible applicability of this emission
fn extract_emission_info<'hir>(
    cx: &LateContext<'hir>,
    args: &'hir [hir::Expr<'hir>],
) -> Option<(String, Option<String>)> {
    let mut lint_name = None;
    let mut applicability = None;

    for arg in args {
        let (arg_ty, _) = walk_ptrs_ty_depth(cx.typeck_results().expr_ty(&arg));

        if match_type(cx, arg_ty, &paths::LINT) {
            // If we found the lint arg, extract the lint name
            if let ExprKind::Path(ref lint_path) = arg.kind {
                lint_name = Some(last_path_segment(lint_path).ident.name);
            }
        } else if match_type(cx, arg_ty, &paths::APPLICABILITY) {
            applicability = resolve_applicability(cx, arg);
        }
    }

    lint_name.map(|lint_name| (sym_to_string(lint_name).to_ascii_lowercase(), applicability))
}

fn extract_complex_emission_info<'hir>(
    cx: &LateContext<'hir>,
    args: &'hir [hir::Expr<'hir>],
) -> Option<(String, Option<String>, bool)> {
    let mut lint_name = None;
    let mut applicability = None;
    let mut multi_span = false;

    for arg in args {
        let (arg_ty, _) = walk_ptrs_ty_depth(cx.typeck_results().expr_ty(&arg));

        if match_type(cx, arg_ty, &paths::LINT) {
            // If we found the lint arg, extract the lint name
            if let ExprKind::Path(ref lint_path) = arg.kind {
                lint_name = Some(last_path_segment(lint_path).ident.name);
            }
        } else if arg_ty.is_closure() {
            if let ExprKind::Closure(_, _, body_id, _, _) = arg.kind {
                let mut visitor = EmissionClosureVisitor::new(cx);
                intravisit::walk_body(&mut visitor, cx.tcx.hir().body(body_id));
                multi_span = visitor.found_multi_span();
                applicability = visitor.complete();
            } else {
                // TODO xfrednet 2021-02-28: linked closures, see: needless_pass_by_value.rs:292
                return None;
            }
        }
    }

    lint_name.map(|lint_name| (sym_to_string(lint_name).to_ascii_lowercase(), applicability, multi_span))
}

/// This function tries to resolve the linked applicability to the given expression.
fn resolve_applicability(cx: &LateContext<'hir>, expr: &'hir hir::Expr<'hir>) -> Option<String> {
    match expr.kind {
        // We can ignore ifs without an else block because those can't be used as an assignment
        ExprKind::If(_con, if_block, Some(else_block)) => {
            let mut visitor = ApplicabilityVisitor::new(cx);
            intravisit::walk_expr(&mut visitor, if_block);
            intravisit::walk_expr(&mut visitor, else_block);
            visitor.complete()
        },
        ExprKind::Match(_expr, arms, _) => {
            let mut visitor = ApplicabilityVisitor::new(cx);
            arms.iter()
                .for_each(|arm| intravisit::walk_expr(&mut visitor, arm.body));
            visitor.complete()
        },
        ExprKind::Loop(block, ..) | ExprKind::Block(block, ..) => {
            let mut visitor = ApplicabilityVisitor::new(cx);
            intravisit::walk_block(&mut visitor, block);
            visitor.complete()
        },
        ExprKind::Path(QPath::Resolved(_, path)) => {
            // direct applicabilities are simple:
            for enum_value in &paths::APPLICABILITY_VALUES {
                if match_path(path, enum_value) {
                    return Some(enum_value[APPLICABILITY_NAME_INDEX].to_string());
                }
            }

            // Values yay
            if let hir::def::Res::Local(local_hir) = path.res {
                if let Some(local) = get_parent_local(cx, local_hir) {
                    if let Some(local_init) = local.init {
                        return resolve_applicability(cx, local_init);
                    }
                }
            }

            // This is true for paths that are linked to function parameters. They might be a bit more work so
            // not today :)
            None
        },
        _ => None,
    }
}

fn get_parent_local(cx: &LateContext<'hir>, hir_id: hir::HirId) -> Option<&'hir hir::Local<'hir>> {
    let map = cx.tcx.hir();

    match map.find(map.get_parent_node(hir_id)) {
        Some(hir::Node::Local(local)) => Some(local),
        Some(hir::Node::Pat(pattern)) => get_parent_local(cx, pattern.hir_id),
        _ => None,
    }
}

/// This visitor finds the highest applicability value in the visited expressions
struct ApplicabilityVisitor<'a, 'hir> {
    cx: &'a LateContext<'hir>,
    /// This is the index of hightest `Applicability` for
    /// `clippy_utils::paths::APPLICABILITY_VALUES`
    applicability_index: Option<usize>,
}

impl<'a, 'hir> ApplicabilityVisitor<'a, 'hir> {
    fn new(cx: &'a LateContext<'hir>) -> Self {
        Self {
            cx,
            applicability_index: None,
        }
    }

    fn add_new_index(&mut self, new_index: usize) {
        self.applicability_index = self
            .applicability_index
            .map_or(new_index, |old_index| old_index.min(new_index))
            .into();
    }

    fn complete(self) -> Option<String> {
        self.applicability_index
            .map(|index| paths::APPLICABILITY_VALUES[index][APPLICABILITY_NAME_INDEX].to_string())
    }
}

impl<'a, 'hir> intravisit::Visitor<'hir> for ApplicabilityVisitor<'a, 'hir> {
    type Map = Map<'hir>;

    fn nested_visit_map(&mut self) -> intravisit::NestedVisitorMap<Self::Map> {
        intravisit::NestedVisitorMap::All(self.cx.tcx.hir())
    }

    fn visit_path(&mut self, path: &hir::Path<'_>, _id: hir::HirId) {
        for (index, enum_value) in paths::APPLICABILITY_VALUES.iter().enumerate() {
            if match_path(path, enum_value) {
                self.add_new_index(index);
                break;
            }
        }
    }
}

/// This visitor finds the highest applicability value in the visited expressions
struct EmissionClosureVisitor<'a, 'hir> {
    cx: &'a LateContext<'hir>,
    /// This is the index of hightest `Applicability` for
    /// `clippy_utils::paths::APPLICABILITY_VALUES`
    applicability_index: Option<usize>,
    suggestion_count: usize,
}

impl<'a, 'hir> EmissionClosureVisitor<'a, 'hir> {
    fn new(cx: &'a LateContext<'hir>) -> Self {
        Self {
            cx,
            applicability_index: None,
            suggestion_count: 0,
        }
    }

    fn add_new_index(&mut self, new_index: usize) {
        self.applicability_index = self
            .applicability_index
            .map_or(new_index, |old_index| old_index.min(new_index))
            .into();
    }

    fn found_multi_span(&self) -> bool {
        self.suggestion_count > 1
    }

    fn complete(self) -> Option<String> {
        self.applicability_index
            .map(|index| paths::APPLICABILITY_VALUES[index][APPLICABILITY_NAME_INDEX].to_string())
    }
}

impl<'a, 'hir> intravisit::Visitor<'hir> for EmissionClosureVisitor<'a, 'hir> {
    type Map = Map<'hir>;

    fn nested_visit_map(&mut self) -> intravisit::NestedVisitorMap<Self::Map> {
        intravisit::NestedVisitorMap::All(self.cx.tcx.hir())
    }

    fn visit_path(&mut self, path: &hir::Path<'_>, _id: hir::HirId) {
        for (index, enum_value) in paths::APPLICABILITY_VALUES.iter().enumerate() {
            if match_path(path, enum_value) {
                self.add_new_index(index);
                break;
            }
        }
    }

    fn visit_expr(&mut self, expr: &'hir hir::Expr<'hir>) {
        match &expr.kind {
            ExprKind::Call(fn_expr, _args) => {
                let found_function = SUGGESTION_FUNCTIONS
                    .iter()
                    .any(|func_path| match_function_call(self.cx, fn_expr, func_path).is_some());
                if found_function {
                    // These functions are all multi part suggestions
                    self.suggestion_count += 2;
                }
            },
            ExprKind::MethodCall(path, _path_span, arg, _arg_span) => {
                let (self_ty, _) = walk_ptrs_ty_depth(self.cx.typeck_results().expr_ty(&arg[0]));
                if match_type(self.cx, self_ty, &paths::DIAGNOSTIC_BUILDER) {
                    let called_method = path.ident.name.as_str().to_string();
                    let found_suggestion =
                        SUGGESTION_DIAGNOSTIC_BUILDER_METHODS
                            .iter()
                            .find_map(|(method_name, is_multi_part)| {
                                if *method_name == called_method {
                                    Some(*is_multi_part)
                                } else {
                                    None
                                }
                            });
                    if let Some(multi_part) = found_suggestion {
                        if multi_part {
                            // two is enough to have it marked as a multipart suggestion
                            self.suggestion_count += 2;
                        } else {
                            self.suggestion_count += 1;
                        }
                    }
                }
            },
            _ => {},
        }

        intravisit::walk_expr(self, expr);
    }
}
