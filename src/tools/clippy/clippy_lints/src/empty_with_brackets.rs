use clippy_utils::attrs::span_contains_cfg;
use clippy_utils::diagnostics::{span_lint_and_then, span_lint_hir_and_then};
use rustc_data_structures::fx::FxIndexMap;
use rustc_errors::Applicability;
use rustc_hir::def::CtorOf;
use rustc_hir::def::DefKind::Ctor;
use rustc_hir::def::Res::Def;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::{Expr, ExprKind, Item, ItemKind, Node, Path, QPath, Variant, VariantData};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::TyCtxt;
use rustc_session::impl_lint_pass;
use rustc_span::Span;

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
        if let ItemKind::Struct(ident, var_data, _) = &item.kind
            && !item.span.from_expansion()
            && has_brackets(var_data)
            && let span_after_ident = item.span.with_lo(ident.span.hi())
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
        // the span of the parentheses/braces
        let span_after_ident = variant.span.with_lo(variant.ident.span.hi());

        if has_no_fields(cx, &variant.data, span_after_ident) {
            match variant.data {
                VariantData::Struct { .. } => {
                    // Empty struct variants can be linted immediately
                    span_lint_and_then(
                        cx,
                        EMPTY_ENUM_VARIANTS_WITH_BRACKETS,
                        span_after_ident,
                        "enum variant has empty brackets",
                        |diagnostic| {
                            diagnostic.span_suggestion_hidden(
                                span_after_ident,
                                "remove the brackets",
                                "",
                                Applicability::MaybeIncorrect,
                            );
                        },
                    );
                },
                VariantData::Tuple(.., local_def_id) => {
                    // Don't lint reachable tuple enums
                    if cx.effective_visibilities.is_reachable(variant.def_id) {
                        return;
                    }
                    if let Some(entry) = self.empty_tuple_enum_variants.get_mut(&local_def_id) {
                        // empty_tuple_enum_variants contains Usage::NoDefinition if the variant was called before the
                        // definition was encountered. Now that there's a definition, convert it
                        // to Usage::Unused.
                        if let Usage::NoDefinition { redundant_use_sites } = entry {
                            *entry = Usage::Unused {
                                redundant_use_sites: redundant_use_sites.clone(),
                            };
                        }
                    } else {
                        self.empty_tuple_enum_variants.insert(
                            local_def_id,
                            Usage::Unused {
                                redundant_use_sites: vec![],
                            },
                        );
                    }
                },
                VariantData::Unit(..) => {},
            }
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        if let Some(def_id) = check_expr_for_enum_as_function(expr) {
            if let Some(parentheses_span) = call_parentheses_span(cx.tcx, expr) {
                // Do not count expressions from macro expansion as a redundant use site.
                if expr.span.from_expansion() {
                    return;
                }
                match self.empty_tuple_enum_variants.get_mut(&def_id) {
                    Some(
                        &mut (Usage::Unused {
                            ref mut redundant_use_sites,
                        }
                        | Usage::NoDefinition {
                            ref mut redundant_use_sites,
                        }),
                    ) => {
                        redundant_use_sites.push(parentheses_span);
                    },
                    None => {
                        // The variant isn't in the IndexMap which means its definition wasn't encountered yet.
                        self.empty_tuple_enum_variants.insert(
                            def_id,
                            Usage::NoDefinition {
                                redundant_use_sites: vec![parentheses_span],
                            },
                        );
                    },
                    _ => {},
                }
            } else {
                // The parentheses are not redundant.
                self.empty_tuple_enum_variants.insert(def_id, Usage::Used);
            }
        }
    }

    fn check_crate_post(&mut self, cx: &LateContext<'_>) {
        for (local_def_id, usage) in &self.empty_tuple_enum_variants {
            // Ignore all variants with Usage::Used or Usage::NoDefinition
            let Usage::Unused { redundant_use_sites } = usage else {
                continue;
            };
            // Attempt to fetch the Variant from LocalDefId.
            let Node::Variant(variant) = cx.tcx.hir_node(
                cx.tcx
                    .local_def_id_to_hir_id(cx.tcx.parent(local_def_id.to_def_id()).expect_local()),
            ) else {
                continue;
            };
            // Span of the parentheses in variant definition
            let span = variant.span.with_lo(variant.ident.span.hi());
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
fn check_expr_for_enum_as_function(expr: &Expr<'_>) -> Option<LocalDefId> {
    if let ExprKind::Path(QPath::Resolved(
        _,
        Path {
            res: Def(Ctor(CtorOf::Variant, _), def_id),
            ..
        },
    )) = expr.kind
    {
        def_id.as_local()
    } else {
        None
    }
}
