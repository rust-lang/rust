use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::macros::{find_assert_args, root_macro_call_first_node};
use clippy_utils::res::MaybeDef as _;
use clippy_utils::source::walk_span_to_context;
use clippy_utils::sugg::Sugg;
use clippy_utils::sym;
use clippy_utils::ty::implements_trait;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, LangItem, UnOp};
use rustc_lint::{LateContext, LateLintPass, LintContext as _};
use rustc_middle::ty::{self, Ty};
use rustc_session::declare_lint_pass;
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Checks assertions that only test whether a supported value is empty.
    ///
    /// The lint handles `assert!` and `debug_assert!` calls on strings, slices, arrays, and `Vec`.
    ///
    /// ### Why is this bad?
    ///
    /// A boolean assertion only reports that the emptiness check failed. It does not show what the
    /// asserted value contained.
    ///
    /// In CI or another remote test service, the failure output tells you that the value was
    /// unexpectedly empty or non-empty, but not which values were present. The next step is often
    /// to reproduce the failure locally, add temporary logging, or change the test so it exposes
    /// the value. That extra investigation can be much slower than fixing the problem from the
    /// original CI failure.
    ///
    /// The emptiness check also commonly appears before a deeper contents assertion:
    ///
    /// ```no_run
    /// # let items = vec!["baz"];
    /// assert!(!items.is_empty());
    /// assert_eq!(items[0], "bar");
    /// ```
    ///
    /// If the first assertion fails, the second assertion never runs, so the failure can hide the
    /// check that would have shown more useful context.
    ///
    /// Instead, compare the value with an empty value using `assert_eq!`, `assert_ne!`,
    /// `debug_assert_eq!`, or `debug_assert_ne!`. These macros print the asserted value on failure.
    ///
    /// ### Known problems
    ///
    /// Printing the asserted value can be undesirable outside tests, especially when the value may
    /// be very large or contain sensitive information. If the assertion failure should not reveal
    /// the value, keep the boolean assertion and allow this lint at that assertion.
    ///
    /// ### Example
    ///
    /// ```no_run
    /// # let items = vec![1, 2, 3];
    /// assert!(items.is_empty());
    /// assert!(!items.is_empty());
    /// ```
    ///
    /// Use instead:
    ///
    /// ```no_run
    /// # let items = vec![1, 2, 3];
    /// assert_eq!(items, [] as [i32; 0]);
    /// assert_ne!(items, [] as [i32; 0]);
    /// ```
    #[clippy::version = "1.98.0"]
    pub ASSERT_IS_EMPTY,
    pedantic,
    "asserting on emptiness without showing the asserted value on failure"
}

declare_lint_pass!(AssertIsEmpty => [ASSERT_IS_EMPTY]);

impl<'tcx> LateLintPass<'tcx> for AssertIsEmpty {
    /// Finds assertion conditions that only test emptiness.
    ///
    /// Matching assertions without a custom message are rewritten as equality or inequality
    /// assertions against an empty value.
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        // `assert!` and `debug_assert!` expand to `if`; skip other expressions before walking the
        // macro backtrace.
        if !matches!(expr.kind, ExprKind::If(..)) {
            return;
        }

        let Some((macro_name, condition, assert_span)) = assert_call(cx, expr) else {
            return;
        };
        let Some((assertion_kind, receiver)) = emptiness_assertion(condition) else {
            return;
        };
        let Some((receiver_suffix, empty_value)) = assertion_suggestion(cx, receiver) else {
            return;
        };

        emit_assertion_suggestion(
            cx,
            macro_name,
            assert_span,
            condition,
            assertion_kind,
            receiver,
            (receiver_suffix, &empty_value),
        );
    }
}

/// Extracts the source-level condition from `assert!` and `debug_assert!`.
///
/// The returned macro name omits the trailing `!` so the diagnostic can build `assert_eq`,
/// `assert_ne`, `debug_assert_eq`, or `debug_assert_ne` from the original macro. Returns `None`
/// for other macros, assertions with a custom message parameter, and conditions from macro
/// expansions, where rewriting the condition span would produce confusing or invalid suggestions.
fn assert_call<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) -> Option<(&'static str, &'tcx Expr<'tcx>, Span)> {
    let macro_call = root_macro_call_first_node(cx, expr)?;
    let macro_name = match cx.tcx.get_diagnostic_name(macro_call.def_id) {
        Some(sym::assert_macro) => "assert",
        Some(sym::debug_assert_macro) => "debug_assert",
        _ => return None,
    };
    let (condition, panic_expn) = find_assert_args(cx, expr, macro_call.expn)?;
    if !panic_expn.is_default_message() {
        return None;
    }

    if condition.span.from_expansion() {
        return None;
    }

    Some((macro_name, condition, macro_call.span))
}

/// Returns the assertion kind and receiver for an emptiness predicate.
///
/// `value.is_empty()` maps to an equality assertion against an empty value. `!value.is_empty()`
/// maps to an inequality assertion. Returns `None` when the condition is neither form.
fn emptiness_assertion<'tcx>(condition: &'tcx Expr<'tcx>) -> Option<(AssertionKind, &'tcx Expr<'tcx>)> {
    if let Some(receiver) = is_empty_receiver(condition) {
        return Some((AssertionKind::Eq, receiver));
    }

    let ExprKind::Unary(UnOp::Not, inner) = condition.kind else {
        return None;
    };

    is_empty_receiver(inner).map(|receiver| (AssertionKind::Ne, receiver))
}

/// Returns the receiver when `expr` is a direct `value.is_empty()` call.
///
/// Returns `None` for other method calls, negated expressions, and non-method expressions.
fn is_empty_receiver<'tcx>(expr: &'tcx Expr<'tcx>) -> Option<&'tcx Expr<'tcx>> {
    let ExprKind::MethodCall(method, receiver, [], _) = expr.kind else {
        return None;
    };
    (method.ident.name == sym::is_empty).then_some(receiver)
}

/// Assertion macro polarity for the replacement assertion.
///
/// The variants are named after the comparison macro suffixes rather than the original predicate
/// shape because suggestions are the only consumer.
#[derive(Clone, Copy, PartialEq, Eq)]
enum AssertionKind {
    /// Use an equality assertion against an empty value.
    Eq,

    /// Use an inequality assertion against an empty value.
    Ne,
}

impl AssertionKind {
    /// Returns the assertion macro suffix for this emptiness predicate.
    ///
    /// Empty checks become equality assertions. Non-empty checks become inequality assertions.
    fn suffix(self) -> &'static str {
        match self {
            Self::Eq => "_eq",
            Self::Ne => "_ne",
        }
    }

    /// Returns the expected state named in the diagnostic.
    fn expected_state(self) -> &'static str {
        match self {
            Self::Eq => "empty",
            Self::Ne => "not empty",
        }
    }
}

/// Builds replacement operands when the resulting assertion is useful.
///
/// The replacement assertion must compile, compare the same value, and print useful failure
/// output. Returns `None` for unsupported collection types and for element types that cannot be
/// printed and compared by the replacement assertion.
fn assertion_suggestion<'tcx>(cx: &LateContext<'tcx>, receiver: &'tcx Expr<'tcx>) -> Option<(&'static str, String)> {
    let receiver_ty = cx.typeck_results().expr_ty(receiver);
    let suggestion = suggestion_for_type(cx, receiver_ty)?;
    if type_is_printable_and_comparable(cx, receiver_ty.peel_refs()) {
        Some(suggestion)
    } else {
        None
    }
}

/// Returns the receiver suffix and empty value for this receiver type.
///
/// Arrays and borrowed vectors are compared through slices because direct comparison with `[]` does
/// not compile for those receiver types. Returns `None` when the receiver has no compact
/// empty-literal comparison.
fn suggestion_for_type<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>) -> Option<(&'static str, String)> {
    match ty.kind() {
        ty::Array(..) => Some((".as_slice()", "[]".to_string())),
        ty::Ref(_, inner, _) if matches!(inner.kind(), ty::Array(..)) => Some((".as_slice()", "[]".to_string())),
        ty::Ref(_, inner, _) if inner.is_diag_item(cx, sym::Vec) => Some((".as_slice()", "[]".to_string())),
        _ => suggestion_for_peeled_type(cx, ty.peel_refs()),
    }
}

/// Returns the empty value for receivers that compare directly after peeling references.
///
/// `String` and `str` compare against `""`. Slices compare against `[]`. `Vec<T>` compares
/// against `[] as [T; 0]` so the empty value carries the element type. Returns `None` for other
/// receiver types.
fn suggestion_for_peeled_type<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>) -> Option<(&'static str, String)> {
    if ty.is_str() || ty.is_lang_item(cx, LangItem::String) {
        Some(("", "\"\"".to_string()))
    } else if matches!(ty.kind(), ty::Slice(..)) {
        Some(("", "[]".to_string()))
    } else if ty.is_diag_item(cx, sym::Vec) {
        Some(("", format!("[] as [{}; 0]", element_type(cx, ty)?)))
    } else {
        None
    }
}

/// Returns whether the replacement assertion has useful failure output.
///
/// Suggestions are limited to cases where the replacement assertion can both compare the value and
/// print it on failure. Strings satisfy this directly; sequence-like values require printable,
/// self-comparable elements. Returns `false` when either trait bound is missing.
fn type_is_printable_and_comparable<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>) -> bool {
    if ty.is_str() || ty.is_lang_item(cx, LangItem::String) {
        return true;
    }

    if let Some(element_ty) = element_type(cx, ty)
        && let Some(debug_trait) = cx.tcx.get_diagnostic_item(sym::Debug)
        && let Some(partial_eq_trait) = cx.tcx.get_diagnostic_item(sym::PartialEq)
    {
        implements_trait(cx, element_ty, debug_trait, &[])
            && implements_trait(cx, element_ty, partial_eq_trait, &[element_ty.into()])
    } else {
        false
    }
}

/// Extracts the element type from supported sequence-like values.
///
/// The element type is used both for trait checks and for the typed empty-array suggestion required
/// by `Vec<T>` suggestions. Returns `None` for non-sequence receiver types.
fn element_type<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>) -> Option<Ty<'tcx>> {
    match ty.kind() {
        ty::Array(ty, _) | ty::Slice(ty) => Some(*ty),
        ty::Adt(_, args) if ty.is_diag_item(cx, sym::Vec) => args.types().next(),
        _ => None,
    }
}

/// Emits the rewrite from a boolean assertion to a comparison assertion.
fn emit_assertion_suggestion(
    cx: &LateContext<'_>,
    macro_name: &str,
    assert_span: Span,
    condition: &Expr<'_>,
    assertion_kind: AssertionKind,
    receiver: &Expr<'_>,
    suggestion: (&str, &str),
) {
    let mut applicability = Applicability::MachineApplicable;
    let receiver_snip =
        Sugg::hir_with_context(cx, receiver, assert_span.ctxt(), "..", &mut applicability).maybe_paren();
    let (receiver_suffix, empty_value) = suggestion;
    let receiver_snip = format!("{receiver_snip}{receiver_suffix}");
    let assertion_suffix = assertion_kind.suffix();
    let expected_state = assertion_kind.expected_state();

    span_lint_and_then(
        cx,
        ASSERT_IS_EMPTY,
        assert_span,
        format!("used `{macro_name}!` to check that a value is {expected_state}"),
        |diag| {
            let macro_name_span = cx.sess().source_map().span_until_char(assert_span, '!');
            let condition_span = walk_span_to_context(condition.span, assert_span.ctxt()).unwrap_or(condition.span);

            diag.multipart_suggestion(
                format!("use `{macro_name}{assertion_suffix}!` to show the value on failure"),
                vec![
                    (macro_name_span.shrink_to_hi(), assertion_suffix.to_string()),
                    (condition_span, format!("{receiver_snip}, {empty_value}")),
                ],
                applicability,
            );
        },
    );
}
