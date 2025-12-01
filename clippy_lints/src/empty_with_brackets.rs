use clippy_utils::attrs::span_contains_cfg;
use clippy_utils::diagnostics::{span_lint_and_then, span_lint_hir_and_then};
use clippy_utils::source::SpanRangeExt;
use clippy_utils::span_contains_non_whitespace;
use rustc_data_structures::fx::{FxIndexMap, IndexEntry};
use rustc_errors::Applicability;
use rustc_hir::def::DefKind::Ctor;
use rustc_hir::def::Res::Def;
use rustc_hir::def::{CtorOf, DefKind};
use rustc_hir::def_id::LocalDefId;
use rustc_hir::{Expr, ExprKind, Item, ItemKind, Node, Pat, PatKind, Path, QPath, Variant, VariantData};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self, TyCtxt};
use rustc_session::impl_lint_pass;
use rustc_span::{BytePos, Span};

declare_clippy_lint! {
    /// ### What it does
    /// Finds structs without fields (a so-called "empty struct") that are declared with brackets.
    ///
    /// ### Why restrict this?
    /// Empty brackets after a struct declaration can be omitted,
    /// and it may be desirable to do so consistently for style.
    ///
    /// However, removing the brackets also introduces a public constant named after the struct,
    /// so this is not just a syntactic simplification but an API change, and adding them back
    /// is a *breaking* API change.
    ///
    /// ### Example
    /// ```no_run
    /// struct Cookie {}
    /// struct Biscuit();
    /// ```
    /// Use instead:
    /// ```no_run
    /// struct Cookie;
    /// struct Biscuit;
    /// ```
    #[clippy::version = "1.62.0"]
    pub EMPTY_STRUCTS_WITH_BRACKETS,
    restriction,
    "finds struct declarations with empty brackets"
}

declare_clippy_lint! {
    /// ### What it does
    /// Finds enum variants without fields that are declared with empty brackets.
    ///
    /// ### Why restrict this?
    /// Empty brackets after a enum variant declaration are redundant and can be omitted,
    /// and it may be desirable to do so consistently for style.
    ///
    /// However, removing the brackets also introduces a public constant named after the variant,
    /// so this is not just a syntactic simplification but an API change, and adding them back
    /// is a *breaking* API change.
    ///
    /// ### Example
    /// ```no_run
    /// enum MyEnum {
    ///     HasData(u8),
    ///     HasNoData(),       // redundant parentheses
    ///     NoneHereEither {}, // redundant braces
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// enum MyEnum {
    ///     HasData(u8),
    ///     HasNoData,
    ///     NoneHereEither,
    /// }
    /// ```
    #[clippy::version = "1.77.0"]
    pub EMPTY_ENUM_VARIANTS_WITH_BRACKETS,
    restriction,
    "finds enum variants with empty brackets"
}

#[derive(Debug)]
enum Usage {
    Unused { redundant_use_sites: Vec<Span> },
    Used,
    NoDefinition { redundant_use_sites: Vec<Span> },
}

#[derive(Default)]
pub struct EmptyWithBrackets {
    // Value holds `Usage::Used` if the empty tuple variant was used as a function
    empty_tuple_enum_variants: FxIndexMap<LocalDefId, Usage>,
}

impl_lint_pass!(EmptyWithBrackets => [EMPTY_STRUCTS_WITH_BRACKETS, EMPTY_ENUM_VARIANTS_WITH_BRACKETS]);

impl LateLintPass<'_> for EmptyWithBrackets {
    fn check_item(&mut self, cx: &LateContext<'_>, item: &Item<'_>) {
        // FIXME: handle `struct $name {}`
        if let ItemKind::Struct(ident, generics, var_data) = &item.kind
            && !item.span.from_expansion()
            && !ident.span.from_expansion()
            && has_brackets(var_data)
            && let span_after_ident = item.span.with_lo(generics.span.hi())
            && has_no_fields(cx, var_data, span_after_ident)
        {
            span_lint_and_then(
                cx,
                EMPTY_STRUCTS_WITH_BRACKETS,
                span_after_ident,
                "found empty brackets on struct declaration",
                |diagnostic| {
                    diagnostic.span_suggestion_hidden(
                        span_after_ident,
                        "remove the brackets",
                        ";",
                        Applicability::Unspecified,
                    );
                },
            );
        }
    }

    fn check_variant(&mut self, cx: &LateContext<'_>, variant: &Variant<'_>) {
        if !variant.span.from_expansion()
            && !variant.ident.span.from_expansion()
            && let span_after_ident = variant.span.with_lo(variant.ident.span.hi())
            && has_no_fields(cx, &variant.data, span_after_ident)
        {
            match variant.data {
                VariantData::Struct { .. } => {
                    self.add_enum_variant(variant.def_id);
                },
                VariantData::Tuple(.., local_def_id) => {
                    // Don't lint reachable tuple enums
                    if cx.effective_visibilities.is_reachable(variant.def_id) {
                        return;
                    }
                    self.add_enum_variant(local_def_id);
                },
                VariantData::Unit(..) => {},
            }
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        if let Some((def_id, mut span)) = check_expr_for_enum_as_function(cx, expr) {
            if span.is_empty()
                && let Some(parentheses_span) = call_parentheses_span(cx.tcx, expr)
            {
                span = parentheses_span;
            }

            if span.is_empty() {
                // The parentheses are not redundant.
                self.empty_tuple_enum_variants.insert(def_id, Usage::Used);
            } else {
                // Do not count expressions from macro expansion as a redundant use site.
                if expr.span.from_expansion() {
                    return;
                }
                self.update_enum_variant_usage(def_id, span);
            }
        }
    }

    fn check_pat(&mut self, cx: &LateContext<'_>, pat: &Pat<'_>) {
        if !pat.span.from_expansion()
            && let Some((def_id, span)) = check_pat_for_enum_as_function(cx, pat)
        {
            self.update_enum_variant_usage(def_id, span);
        }
    }

    fn check_crate_post(&mut self, cx: &LateContext<'_>) {
        for (&local_def_id, usage) in &self.empty_tuple_enum_variants {
            // Ignore all variants with Usage::Used or Usage::NoDefinition
            let Usage::Unused { redundant_use_sites } = usage else {
                continue;
            };

            // Attempt to fetch the Variant from LocalDefId.
            let variant = if let Node::Variant(variant) = cx.tcx.hir_node_by_def_id(local_def_id) {
                variant
            } else if let Node::Variant(variant) = cx.tcx.hir_node_by_def_id(cx.tcx.local_parent(local_def_id)) {
                variant
            } else {
                continue;
            };

            // Span of the parentheses in variant definition
            let span = variant.span.with_lo(variant.ident.span.hi());
            let span_inner = span
                .with_lo(SpanRangeExt::trim_start(span, cx).start + BytePos(1))
                .with_hi(span.hi() - BytePos(1));
            if span_contains_non_whitespace(cx, span_inner, false) {
                continue;
            }
            span_lint_hir_and_then(
                cx,
                EMPTY_ENUM_VARIANTS_WITH_BRACKETS,
                variant.hir_id,
                span,
                "enum variant has empty brackets",
                |diagnostic| {
                    if redundant_use_sites.is_empty() {
                        // If there's no redundant use sites, the definition is the only place to modify.
                        diagnostic.span_suggestion_hidden(
                            span,
                            "remove the brackets",
                            "",
                            Applicability::MaybeIncorrect,
                        );
                    } else {
                        let mut parentheses_spans: Vec<_> =
                            redundant_use_sites.iter().map(|span| (*span, String::new())).collect();
                        parentheses_spans.push((span, String::new()));
                        diagnostic.multipart_suggestion(
                            "remove the brackets",
                            parentheses_spans,
                            Applicability::MaybeIncorrect,
                        );
                    }
                },
            );
        }
    }
}

impl EmptyWithBrackets {
    fn add_enum_variant(&mut self, local_def_id: LocalDefId) {
        self.empty_tuple_enum_variants
            .entry(local_def_id)
            .and_modify(|entry| {
                // empty_tuple_enum_variants contains Usage::NoDefinition if the variant was called before
                // the definition was encountered. Now that there's a
                // definition, convert it to Usage::Unused.
                if let Usage::NoDefinition { redundant_use_sites } = entry {
                    *entry = Usage::Unused {
                        redundant_use_sites: redundant_use_sites.clone(),
                    };
                }
            })
            .or_insert_with(|| Usage::Unused {
                redundant_use_sites: vec![],
            });
    }

    fn update_enum_variant_usage(&mut self, def_id: LocalDefId, parentheses_span: Span) {
        match self.empty_tuple_enum_variants.entry(def_id) {
            IndexEntry::Occupied(mut e) => {
                if let Usage::Unused { redundant_use_sites } | Usage::NoDefinition { redundant_use_sites } = e.get_mut()
                {
                    redundant_use_sites.push(parentheses_span);
                }
            },
            IndexEntry::Vacant(e) => {
                // The variant isn't in the IndexMap which means its definition wasn't encountered yet.
                e.insert(Usage::NoDefinition {
                    redundant_use_sites: vec![parentheses_span],
                });
            },
        }
    }
}

fn has_brackets(var_data: &VariantData<'_>) -> bool {
    !matches!(var_data, VariantData::Unit(..))
}

fn has_no_fields(cx: &LateContext<'_>, var_data: &VariantData<'_>, braces_span: Span) -> bool {
    var_data.fields().is_empty() &&
    // there might still be field declarations hidden from the AST
    // (conditionally compiled code using #[cfg(..)])
    !span_contains_cfg(cx, braces_span)
}

// If expression HIR ID and callee HIR ID are same, returns the span of the parentheses, else,
// returns None.
fn call_parentheses_span(tcx: TyCtxt<'_>, expr: &Expr<'_>) -> Option<Span> {
    if let Node::Expr(parent) = tcx.parent_hir_node(expr.hir_id)
        && let ExprKind::Call(callee, ..) = parent.kind
        && callee.hir_id == expr.hir_id
    {
        Some(parent.span.with_lo(expr.span.hi()))
    } else {
        None
    }
}

// Returns the LocalDefId of the variant being called as a function if it exists.
fn check_expr_for_enum_as_function(cx: &LateContext<'_>, expr: &Expr<'_>) -> Option<(LocalDefId, Span)> {
    match expr.kind {
        ExprKind::Path(QPath::Resolved(
            _,
            Path {
                res: Def(Ctor(CtorOf::Variant, _), def_id),
                span,
                ..
            },
        )) => def_id.as_local().map(|id| (id, span.with_lo(expr.span.hi()))),
        ExprKind::Struct(qpath, ..)
            if let Def(DefKind::Variant, mut def_id) = cx.typeck_results().qpath_res(qpath, expr.hir_id) =>
        {
            let ty = cx.tcx.type_of(def_id).instantiate_identity();
            if let ty::FnDef(ctor_def_id, _) = ty.kind() {
                def_id = *ctor_def_id;
            }

            def_id.as_local().map(|id| (id, qpath.span().with_lo(expr.span.hi())))
        },
        _ => None,
    }
}

fn check_pat_for_enum_as_function(cx: &LateContext<'_>, pat: &Pat<'_>) -> Option<(LocalDefId, Span)> {
    match pat.kind {
        PatKind::TupleStruct(qpath, ..)
            if let Def(Ctor(CtorOf::Variant, _), def_id) = cx.typeck_results().qpath_res(&qpath, pat.hir_id) =>
        {
            def_id.as_local().map(|id| (id, qpath.span().with_lo(pat.span.hi())))
        },
        PatKind::Struct(qpath, ..)
            if let Def(DefKind::Variant, mut def_id) = cx.typeck_results().qpath_res(&qpath, pat.hir_id) =>
        {
            let ty = cx.tcx.type_of(def_id).instantiate_identity();
            if let ty::FnDef(ctor_def_id, _) = ty.kind() {
                def_id = *ctor_def_id;
            }

            def_id.as_local().map(|id| (id, qpath.span().with_lo(pat.span.hi())))
        },
        _ => None,
    }
}
