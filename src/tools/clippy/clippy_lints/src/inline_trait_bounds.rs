use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::{HasSession, snippet};
use rustc_ast::NodeId;
use rustc_ast::ast::{Fn, FnRetTy, GenericParam, GenericParamKind};
use rustc_ast::visit::{FnCtxt, FnKind};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Enforce that `where` bounds are used for all trait bounds.
    ///
    /// ### Why restrict this?
    /// Enforce a single style throughout a codebase.
    /// Avoid uncertainty about whether a bound should be inline
    /// or out-of-line (i.e. a where bound).
    /// Avoid complex inline bounds, which could make a function declaration more difficult to read.
    ///
    /// ### Known limitations
    /// Only lints functions and method declararions. Bounds on structs, enums,
    /// and impl blocks are not yet covered.
    ///
    /// ### Example
    /// ```no_run
    /// fn foo<T: Clone>() {}
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// fn foo<T>() where T: Clone {}
    /// ```
    #[clippy::version = "1.97.0"]
    pub INLINE_TRAIT_BOUNDS,
    restriction,
    "enforce that `where` bounds are used for all trait bounds"
}

declare_lint_pass!(InlineTraitBounds => [INLINE_TRAIT_BOUNDS]);

impl EarlyLintPass for InlineTraitBounds {
    fn check_fn(&mut self, cx: &EarlyContext<'_>, kind: FnKind<'_>, _: Span, _: NodeId) {
        let FnKind::Fn(ctxt, _vis, f) = kind else {
            return;
        };

        // Skip foreign functions (extern "C" etc.)
        if !matches!(ctxt, FnCtxt::Free | FnCtxt::Assoc(..)) {
            return;
        }

        if f.sig.span.in_external_macro(cx.sess().source_map()) {
            return;
        }

        lint_fn(cx, f);
    }
}

fn lint_fn(cx: &EarlyContext<'_>, f: &Fn) {
    let generics = &f.generics;
    let offenders: Vec<&GenericParam> = generics
        .params
        .iter()
        .filter(|param| {
            !param.bounds.is_empty() && matches!(param.kind, GenericParamKind::Lifetime | GenericParamKind::Type { .. })
        })
        .collect();
    if offenders.is_empty() {
        return;
    }

    let predicates = offenders
        .iter()
        .map(|param| build_predicate_text(cx, param))
        .collect::<Vec<_>>();

    let mut edits = Vec::new();

    for param in offenders {
        if let Some(colon) = param.colon_span {
            let remove_span = colon.to(param.bounds.last().unwrap().span());

            edits.push((remove_span, String::new()));
        }
    }

    let predicate_text = predicates.join(", ");

    let where_clause = &generics.where_clause;
    if where_clause.has_where_token {
        let (insert_at, suffix) = if let Some(last_pred) = where_clause.predicates.last() {
            // existing `where` with predicates: append after last predicate
            (last_pred.span.shrink_to_hi(), format!(", {predicate_text}"))
        } else {
            // `where` token present but empty predicate list
            (where_clause.span.shrink_to_hi(), format!(" {predicate_text}"))
        };

        edits.push((insert_at, suffix));
    } else {
        let insert_at = match &f.sig.decl.output {
            FnRetTy::Default(span) => span.shrink_to_lo(),
            FnRetTy::Ty(ty) => ty.span.shrink_to_hi(),
        };
        edits.push((insert_at, format!(" where {predicate_text}")));
    }

    span_lint_and_then(
        cx,
        INLINE_TRAIT_BOUNDS,
        generics.span,
        "inline trait bounds used",
        |diag| {
            diag.multipart_suggestion(
                "move bounds to a `where` clause",
                edits,
                Applicability::MachineApplicable,
            );
        },
    );
}

fn build_predicate_text(cx: &EarlyContext<'_>, param: &GenericParam) -> String {
    // bounds is guaranteed non-empty by the filter in `lint_fn`
    let first = param.bounds.first().unwrap();
    let last = param.bounds.last().unwrap();

    let bounds_span = first.span().to(last.span());

    let lhs = snippet(cx, param.ident.span, "..");

    let rhs = snippet(cx, bounds_span, "..");

    format!("{lhs}: {rhs}")
}
