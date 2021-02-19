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
use rustc_middle::ty::{BorrowKind, Ty};
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
    get_enclosing_body, get_parent_expr_for_hir, last_path_segment, match_function_call, match_qpath, match_type,
    path_to_local_id, paths, span_lint, walk_ptrs_ty_depth, get_parent_expr
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
        if_chain! {
            if let Some(local_ty) = get_local_type(cx, local);
            if match_type(cx, local_ty, &paths::APPLICABILITY);
            if let Some(body) = get_enclosing_body(cx, local.hir_id);
            then {
                // TODO xFrednet: 2021-02-19: Remove debug code
                let span = SerializableSpan::from_span(cx, local.span);
                let local_str = crate::utils::snippet(cx, local.span, "_");
                log_to_file(&format!("{} -- {}\n", local_str, span));
                
                let value_hir_id = local.pat.hir_id;
                let mut tracker = ValueTracker::new(cx, value_hir_id);
                if let Some(init_expr) = local.init {
                    tracker.process_assign_expr(init_expr)
                }

                // TODO xFrednet 2021-02-18: Support nested bodies
                // Note: The `ExprUseVisitor` only searches though one body, this means that values
                // references in nested bodies like closures are not found by this simple visitor.
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

                log_to_file(&format!("{:?}\n", tracker.value_mutations));
            }
        }
    }
}

fn get_local_type<'a>(cx: &'a LateContext<'_>, local: &'a hir::Local<'_>) -> Option<Ty<'a>> {
    // TODO xFrednet 2021-02-14: support nested applicability (only in tuples)
    if let Some(tc) = cx.maybe_typeck_results() {
        if let Some(ty) = local.ty {
            return Some(hir_ty_to_ty(cx.tcx, ty));
        } else if let Some(init) = local.init {
            return Some(tc.expr_ty(init));
        }
    }

    None
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

#[allow(dead_code)]
struct ValueTracker<'a, 'hir> {
    cx: &'a LateContext<'hir>,
    value_hir_id: hir::HirId,
    value_mutations: Vec<ApplicabilityModifier<'hir>>,
}

impl<'a, 'hir> ValueTracker<'a, 'hir> {
    fn new(cx: &'a LateContext<'hir>, value_hir_id: hir::HirId) -> Self {
        Self {
            cx,
            value_hir_id,
            value_mutations: Vec::new(),
        }
    }

    fn is_value_expr(&self, expr_id: hir::HirId) -> bool {
        match self.cx.tcx.hir().find(expr_id) {
            Some(hir::Node::Expr(expr)) => path_to_local_id(expr, self.value_hir_id),
            _ => false,
        }
    }

    /// This function extracts possible `ApplicabilityModifier` from an assign statement like this:
    ///
    /// ```rust, ignore
    /// //          vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv The expression to process
    /// let value = Applicability::MachineApplicable;
    /// ```
    fn process_assign_expr(&mut self, expr: &'hir hir::Expr<'hir>) {
        // This is a bit more complicated. I'll therefor settle on the simple solution of
        // simplifying the cases we support.
        match &expr.kind {
            hir::ExprKind::Call(func_expr, ..) => {
                // We only deal with resolved paths as this is the usual case. Other expression kinds like closures
                // etc. are hard to track but might be a worthy improvement in the future
                if let hir::ExprKind::Path(hir::QPath::Resolved(_, path)) = func_expr.kind {
                    self.value_mutations.push(ApplicabilityModifier::Producer(path));
                } else {
                    let msg = format!(
                        "Unsupported assign Call expression at: {}",
                        SerializableSpan::from_span(self.cx, func_expr.span)
                    );
                    self.value_mutations.push(ApplicabilityModifier::Unknown(msg));
                }
            },
            hir::ExprKind::MethodCall(..) => {
                let msg = format!(
                    "Unsupported assign MethodCall expression at: {}",
                    SerializableSpan::from_span(self.cx, expr.span)
                );
                self.value_mutations.push(ApplicabilityModifier::Unknown(msg));
            },
            // We can ignore ifs without an else block because those can't be used as an assignment
            hir::ExprKind::If(_con, if_block, Some(else_block)) => {
                self.process_assign_expr(if_block);
                self.process_assign_expr(else_block);
            },
            hir::ExprKind::Match(_expr, arms, _) => {
                for arm in *arms {
                    self.process_assign_expr(arm.body);
                }
            },
            hir::ExprKind::Loop(block, ..) | hir::ExprKind::Block(block, ..) => {
                if let Some(block_expr) = block.expr {
                    self.process_assign_expr(block_expr);
                }
            },
            hir::ExprKind::Path(path) => {
                for enum_value in &paths::APPLICABILITY_VALUES {
                    if match_qpath(path, enum_value) {
                        self.value_mutations
                            .push(ApplicabilityModifier::ConstValue(enum_value[2].to_string()));
                    }
                }
            },
            // hir::ExprKind::Field(expr, ident) => not supported
            // hir::ExprKind::Index(expr, expr) => not supported
            _ => {
                let msg = format!(
                    "Unexpected assign expression at: {}",
                    SerializableSpan::from_span(self.cx, expr.span)
                );
                self.value_mutations.push(ApplicabilityModifier::Unknown(msg));
            },
        }
    }

    fn process_borrow_expr(&mut self, access_hir_id: hir::HirId) {
        let borrower: &rustc_hir::Expr<'_>;
        if let Some(addr_of_expr) = get_parent_expr_for_hir(self.cx, access_hir_id) {
            if let Some(borrower_expr) = get_parent_expr(self.cx, addr_of_expr) {
                borrower = borrower_expr
            } else {
                return;
            }
        } else {
            return;
        }

        match &borrower.kind {
            hir::ExprKind::Call(func_expr, ..) => {
                // We only deal with resolved paths as this is the usual case. Other expression kinds like closures
                // etc. are hard to track but might be a worthy improvement in the future
                if let hir::ExprKind::Path(hir::QPath::Resolved(_, path)) = func_expr.kind {
                    self.value_mutations.push(ApplicabilityModifier::Modifier(path));
                } else {
                    let msg = format!(
                        "Unsupported borrow in Call at: {}",
                        SerializableSpan::from_span(self.cx, func_expr.span)
                    );
                    self.value_mutations.push(ApplicabilityModifier::Unknown(msg));
                }
            },
            hir::ExprKind::MethodCall(..) => {
                let msg = format!(
                    "Unsupported borrow in MethodCall at: {}",
                    SerializableSpan::from_span(self.cx, borrower.span)
                );
                self.value_mutations.push(ApplicabilityModifier::Unknown(msg));
            },
            _ => {
                let msg = format!(
                    "Unexpected borrow at: {}",
                    SerializableSpan::from_span(self.cx, borrower.span)
                );
                self.value_mutations.push(ApplicabilityModifier::Unknown(msg));
            },
        }
    }
}

impl<'a, 'hir> Delegate<'hir> for ValueTracker<'a, 'hir> {
    fn consume(&mut self, _place_with_id: &PlaceWithHirId<'hir>, expr_id: hir::HirId, _: ConsumeMode) {
        if self.is_value_expr(expr_id) {
            // TODO xFrednet 2021-02-17: Check if lint emission and extract lint ID
            if let Some(hir::Node::Expr(expr)) = self.cx.tcx.hir().find(expr_id) {
                let span = SerializableSpan::from_span(self.cx, expr.span);
                log_to_file(&format!("- consume {}\n", span));
            }
        }
    }

    fn borrow(&mut self, _place_with_id: &PlaceWithHirId<'hir>, expr_id: hir::HirId, bk: BorrowKind) {
        if self.is_value_expr(expr_id) {
            if let BorrowKind::MutBorrow = bk {
                self.process_borrow_expr(expr_id);
            }
        }
    }

    fn mutate(&mut self, _assignee_place: &PlaceWithHirId<'hir>, expr_id: hir::HirId) {
        if_chain! {
            if self.is_value_expr(expr_id);
            if let Some(expr) = get_parent_expr_for_hir(self.cx, expr_id);
            if let hir::ExprKind::Assign(_value_expr, assign_expr, ..) = expr.kind;
            then {
                self.process_assign_expr(assign_expr);
            }
        }
    }
}

/// The life of a value in Rust is a true adventure. These are the corner stones of such a
/// fairy tale. Let me introduce you to the possible stepping stones a value might have in
/// in our crazy word:
#[derive(Debug)]
#[allow(dead_code)]
enum ApplicabilityModifier<'hir> {
    Unknown(String),
    /// A simple constant value.
    ///
    /// This is the actual character of a value. It's baseline. This only defines where the value
    /// started. As in real life it can still change and fully decide who it wants to be.
    ConstValue(String),
    /// A producer is a function that returns an applicability value.
    ///
    /// This is the heritage of this value. This value comes from a long family tree and is not
    /// just a black piece of paper. The evaluation of this stepping stone needs additional
    /// context. We therefore only add a reference. This reference will later be used to ask
    /// the librarian about the possible initial character that this value might have.
    Producer(&'hir hir::Path<'hir>),
    /// A modifier that takes the given applicability and might modify it
    ///
    /// What would an RPG be without it's NPCs. The special thing about modifiers is that they can
    /// be actively interested in the story of the value and might make decisions based on the
    /// character of this hero. This means that a modifier doesn't just force its way into the life
    /// of our hero but it actually asks him how he's been. The possible modification is a result
    /// of the situation.
    ///
    /// Take this part of our heroes life very seriously!
    Modifier(&'hir hir::Path<'hir>),
    /// The actual emission of a lint
    ///
    /// All good things must come to an end. Even the life of your awesome applicability hero. He
    /// was the bravest soul that has ever wondered this earth. Songs will be written about his
    /// heroic deeds. Castles will be named after him and the world how we know it will never be
    /// the same!
    ///
    /// Is this a happy ending? Did he archive what he wanted in his life? Yes, YES, he has lived a
    /// life and he will continue to live in all the lint suggestions that can be applied or just
    /// displayed by Clippy. He might be no more, but his legacy will serve generations to come.
    LintEmit(LintEmission),
}

#[derive(Debug)]
struct LintEmission {
    lint: String,
    is_multi_line_sugg: bool,
}
