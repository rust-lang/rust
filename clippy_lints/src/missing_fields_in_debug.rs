use std::ops::ControlFlow;

use clippy_utils::{
    diagnostics::span_lint_and_then,
    paths,
    ty::match_type,
    visitors::{for_each_expr, Visitable},
};
use rustc_ast::LitKind;
use rustc_data_structures::fx::FxHashSet;
use rustc_hir::{
    def::{DefKind, Res},
    Expr, ImplItemKind, MatchSource, Node,
};
use rustc_hir::{Block, PatKind};
use rustc_hir::{ExprKind, Impl, ItemKind, QPath, TyKind};
use rustc_hir::{ImplItem, Item, VariantData};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::TypeckResults;
use rustc_middle::ty::{EarlyBinder, Ty};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{sym, Span, Symbol};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for manual [`core::fmt::Debug`](https://doc.rust-lang.org/core/fmt/trait.Debug.html) implementations that do not use all fields.
    ///
    /// ### Why is this bad?
    /// A common mistake is to forget to update manual `Debug` implementations when adding a new field
    /// to a struct or a new variant to an enum.
    ///
    /// At the same time, it also acts as a style lint to suggest using [`core::fmt::DebugStruct::finish_non_exhaustive`](https://doc.rust-lang.org/core/fmt/struct.DebugStruct.html#method.finish_non_exhaustive)
    /// for the times when the user intentionally wants to leave out certain fields (e.g. to hide implementation details).
    ///
    /// ### Known problems
    /// This lint works based on the `DebugStruct` helper types provided by the `Formatter`,
    /// so this won't detect `Debug` impls that use the `write!` macro.
    /// Oftentimes there is more logic to a `Debug` impl if it uses `write!` macro, so it tries
    /// to be on the conservative side and not lint in those cases in an attempt to prevent false positives.
    ///
    /// This lint also does not look through function calls, so calling `.field(self.as_slice())` for example
    /// does not consider fields used inside of `as_slice()` as used by the `Debug` impl.
    ///
    /// Lastly, it also ignores tuple structs as their `DebugTuple` formatter does not have a `finish_non_exhaustive`
    /// method.
    ///
    /// ### Example
    /// ```rust
    /// use std::fmt;
    /// struct Foo {
    ///     data: String,
    ///     // implementation detail
    ///     hidden_data: i32
    /// }
    /// impl fmt::Debug for Foo {
    ///     fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         formatter
    ///             .debug_struct("Foo")
    ///             .field("data", &self.data)
    ///             .finish()
    ///     }
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// use std::fmt;
    /// struct Foo {
    ///     data: String,
    ///     // implementation detail
    ///     hidden_data: i32
    /// }
    /// impl fmt::Debug for Foo {
    ///     fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         formatter
    ///             .debug_struct("Foo")
    ///             .field("data", &self.data)
    ///             .finish_non_exhaustive()
    ///     }
    /// }
    /// ```
    #[clippy::version = "1.70.0"]
    pub MISSING_FIELDS_IN_DEBUG,
    pedantic,
    "missing fields in manual `Debug` implementation"
}
declare_lint_pass!(MissingFieldsInDebug => [MISSING_FIELDS_IN_DEBUG]);

fn report_lints(cx: &LateContext<'_>, span: Span, span_notes: Vec<(Span, &'static str)>) {
    span_lint_and_then(
        cx,
        MISSING_FIELDS_IN_DEBUG,
        span,
        "manual `Debug` impl does not include all fields",
        |diag| {
            for (span, note) in span_notes {
                diag.span_note(span, note);
            }
            diag.help("consider including all fields in this `Debug` impl")
                .help("consider calling `.finish_non_exhaustive()` if you intend to ignore fields");
        },
    );
}

/// Checks if we should lint in a block of code
///
/// The way we check for this condition is by checking if there is
/// a call to `Formatter::debug_struct` but no call to `.finish_non_exhaustive()`.
fn should_lint<'tcx>(
    cx: &LateContext<'tcx>,
    typeck_results: &TypeckResults<'tcx>,
    block: impl Visitable<'tcx>,
) -> bool {
    // Is there a call to `DebugStruct::finish_non_exhaustive`? Don't lint if there is.
    let mut has_finish_non_exhaustive = false;
    // Is there a call to `DebugStruct::debug_struct`? Do lint if there is.
    let mut has_debug_struct = false;

    for_each_expr(block, |expr| {
        if let ExprKind::MethodCall(path, recv, ..) = &expr.kind {
            let recv_ty = typeck_results.expr_ty(recv).peel_refs();

            if path.ident.name == sym::debug_struct && match_type(cx, recv_ty, &paths::FORMATTER) {
                has_debug_struct = true;
            } else if path.ident.name == sym!(finish_non_exhaustive) && match_type(cx, recv_ty, &paths::DEBUG_STRUCT) {
                has_finish_non_exhaustive = true;
            }
        }
        ControlFlow::<!, _>::Continue(())
    });

    !has_finish_non_exhaustive && has_debug_struct
}

/// Checks if the given expression is a call to `DebugStruct::field`
/// and the first argument to it is a string literal and if so, returns it
///
/// Example: `.field("foo", ....)` returns `Some("foo")`
fn as_field_call<'tcx>(
    cx: &LateContext<'tcx>,
    typeck_results: &TypeckResults<'tcx>,
    expr: &Expr<'_>,
) -> Option<Symbol> {
    if let ExprKind::MethodCall(path, recv, [debug_field, _], _) = &expr.kind
        && let recv_ty = typeck_results.expr_ty(recv).peel_refs()
        && match_type(cx, recv_ty, &paths::DEBUG_STRUCT)
        && path.ident.name == sym::field
        && let ExprKind::Lit(lit) = &debug_field.kind
        && let LitKind::Str(sym, ..) = lit.node
    {
        Some(sym)
    } else {
        None
    }
}

/// Attempts to find unused fields assuming that the item is a struct
fn check_struct<'tcx>(
    cx: &LateContext<'tcx>,
    typeck_results: &TypeckResults<'tcx>,
    block: &'tcx Block<'tcx>,
    self_ty: Ty<'tcx>,
    item: &'tcx Item<'tcx>,
    data: &VariantData<'_>,
) {
    // Is there a "direct" field access anywhere (i.e. self.foo)?
    // We don't want to lint if there is not, because the user might have
    // a newtype struct and use fields from the wrapped type only.
    let mut has_direct_field_access = false;
    let mut field_accesses = FxHashSet::default();

    for_each_expr(block, |expr| {
        if let ExprKind::Field(target, ident) = expr.kind
            && let target_ty = typeck_results.expr_ty_adjusted(target).peel_refs()
            && target_ty == self_ty
        {
            field_accesses.insert(ident.name);
            has_direct_field_access = true;
        } else if let Some(sym) = as_field_call(cx, typeck_results, expr) {
            field_accesses.insert(sym);
        }
        ControlFlow::<!, _>::Continue(())
    });

    let span_notes = data
        .fields()
        .iter()
        .filter_map(|field| {
            let EarlyBinder(field_ty) = cx.tcx.type_of(field.def_id);
            if field_accesses.contains(&field.ident.name) || field_ty.is_phantom_data() {
                None
            } else {
                Some((field.span, "this field is unused"))
            }
        })
        .collect::<Vec<_>>();

    // only lint if there's also at least one direct field access to allow patterns
    // where one might have a newtype struct and uses fields from the wrapped type
    if !span_notes.is_empty() && has_direct_field_access {
        report_lints(cx, item.span, span_notes);
    }
}

/// Attempts to find unused fields in variants assuming that
/// the item is an enum.
///
/// Currently, only simple cases are detected where the user
/// matches on `self` and calls `debug_struct` inside of the arms
fn check_enum<'tcx>(
    cx: &LateContext<'tcx>,
    typeck_results: &TypeckResults<'tcx>,
    block: &'tcx Block<'tcx>,
    self_ty: Ty<'tcx>,
    item: &'tcx Item<'tcx>,
) {
    let Some(arms) = for_each_expr(block, |expr| {
        if let ExprKind::Match(val, arms, MatchSource::Normal) = expr.kind
            && let match_ty = typeck_results.expr_ty_adjusted(val).peel_refs()
            && match_ty == self_ty
        {
            ControlFlow::Break(arms)
        } else {
            ControlFlow::Continue(())
        }
    }) else {
        return;
    };

    let mut span_notes = Vec::new();

    for arm in arms {
        if !should_lint(cx, typeck_results, arm.body) {
            continue;
        }

        arm.pat.walk_always(|pat| match pat.kind {
            PatKind::Wild => span_notes.push((pat.span, "unused field here due to wildcard `_`")),
            PatKind::Tuple(_, rest) | PatKind::TupleStruct(.., rest) if rest.as_opt_usize().is_some() => {
                span_notes.push((pat.span, "more unused fields here due to rest pattern `..`"));
            },
            PatKind::Struct(.., true) => {
                span_notes.push((pat.span, "more unused fields here due to rest pattern `..`"));
            },
            _ => {},
        });

        let mut field_accesses = FxHashSet::default();
        let mut check_field_access = |sym, expr| {
            if !typeck_results.expr_ty(expr).is_phantom_data() {
                arm.pat.each_binding(|_, _, _, pat_ident| {
                    if sym == pat_ident.name {
                        field_accesses.insert(pat_ident);
                    }
                });
            }
        };

        for_each_expr(arm.body, |expr| {
            if let ExprKind::Path(QPath::Resolved(_, path)) = expr.kind && let Some(segment) = path.segments.first()
            {
                check_field_access(segment.ident.name, expr);
            } else if let Some(sym) = as_field_call(cx, typeck_results, expr) {
                check_field_access(sym, expr);
            }
            ControlFlow::<!, _>::Continue(())
        });

        arm.pat.each_binding(|_, _, span, pat_ident| {
            if !field_accesses.contains(&pat_ident) {
                span_notes.push((span, "the field referenced by this binding is unused"));
            }
        });
    }

    if !span_notes.is_empty() {
        report_lints(cx, item.span, span_notes);
    }
}

impl<'tcx> LateLintPass<'tcx> for MissingFieldsInDebug {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx rustc_hir::Item<'tcx>) {
        // is this an `impl Debug for X` block?
        if let ItemKind::Impl(Impl { of_trait: Some(trait_ref), self_ty, items, .. }) = item.kind
            && let Res::Def(DefKind::Trait, trait_def_id) = trait_ref.path.res
            && let TyKind::Path(QPath::Resolved(_, self_path)) = &self_ty.kind
            && cx.match_def_path(trait_def_id, &[sym::core, sym::fmt, sym::Debug])
            // don't trigger if this impl was derived
            && !cx.tcx.has_attr(item.owner_id, sym::automatically_derived)
            && !item.span.from_expansion()
            // find `Debug::fmt` function
            && let Some(fmt_item) = items.iter().find(|i| i.ident.name == sym::fmt)
            && let ImplItem { kind: ImplItemKind::Fn(_, body_id), .. } = cx.tcx.hir().impl_item(fmt_item.id)
            && let body = cx.tcx.hir().body(*body_id)
            && let ExprKind::Block(block, _) = body.value.kind
            // inspect `self`
            && let self_ty = cx.tcx.type_of(self_path.res.def_id()).0.peel_refs()
            && let Some(self_adt) = self_ty.ty_adt_def()
            && let Some(self_def_id) = self_adt.did().as_local()
            && let Some(Node::Item(self_item)) = cx.tcx.hir().find_by_def_id(self_def_id)
            // NB: can't call cx.typeck_results() as we are not in a body
            && let typeck_results = cx.tcx.typeck_body(*body_id)
            && should_lint(cx, typeck_results, block)
        {
            match &self_item.kind {
                ItemKind::Struct(data, _) => check_struct(cx, typeck_results, block, self_ty, item, data),
                ItemKind::Enum(..) => check_enum(cx, typeck_results, block, self_ty, item),
                _ => {}
            }
        }
    }
}
