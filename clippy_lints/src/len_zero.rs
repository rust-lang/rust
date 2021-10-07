use clippy_utils::diagnostics::{span_lint, span_lint_and_sugg, span_lint_and_then};
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{get_item_name, get_parent_as_impl, is_lint_allowed};
use if_chain::if_chain;
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::def_id::DefIdSet;
use rustc_hir::{
    def_id::DefId, AssocItemKind, BinOpKind, Expr, ExprKind, FnRetTy, ImplItem, ImplItemKind, ImplicitSelfKind, Item,
    ItemKind, Mutability, Node, TraitItemRef, TyKind,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self, AssocKind, FnSig, Ty, TyS};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{
    source_map::{Span, Spanned, Symbol},
    symbol::sym,
};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for getting the length of something via `.len()`
    /// just to compare to zero, and suggests using `.is_empty()` where applicable.
    ///
    /// ### Why is this bad?
    /// Some structures can answer `.is_empty()` much faster
    /// than calculating their length. So it is good to get into the habit of using
    /// `.is_empty()`, and having it is cheap.
    /// Besides, it makes the intent clearer than a manual comparison in some contexts.
    ///
    /// ### Example
    /// ```ignore
    /// if x.len() == 0 {
    ///     ..
    /// }
    /// if y.len() != 0 {
    ///     ..
    /// }
    /// ```
    /// instead use
    /// ```ignore
    /// if x.is_empty() {
    ///     ..
    /// }
    /// if !y.is_empty() {
    ///     ..
    /// }
    /// ```
    pub LEN_ZERO,
    style,
    "checking `.len() == 0` or `.len() > 0` (or similar) when `.is_empty()` could be used instead"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for items that implement `.len()` but not
    /// `.is_empty()`.
    ///
    /// ### Why is this bad?
    /// It is good custom to have both methods, because for
    /// some data structures, asking about the length will be a costly operation,
    /// whereas `.is_empty()` can usually answer in constant time. Also it used to
    /// lead to false positives on the [`len_zero`](#len_zero) lint – currently that
    /// lint will ignore such entities.
    ///
    /// ### Example
    /// ```ignore
    /// impl X {
    ///     pub fn len(&self) -> usize {
    ///         ..
    ///     }
    /// }
    /// ```
    pub LEN_WITHOUT_IS_EMPTY,
    style,
    "traits or impls with a public `len` method but no corresponding `is_empty` method"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for comparing to an empty slice such as `""` or `[]`,
    /// and suggests using `.is_empty()` where applicable.
    ///
    /// ### Why is this bad?
    /// Some structures can answer `.is_empty()` much faster
    /// than checking for equality. So it is good to get into the habit of using
    /// `.is_empty()`, and having it is cheap.
    /// Besides, it makes the intent clearer than a manual comparison in some contexts.
    ///
    /// ### Example
    ///
    /// ```ignore
    /// if s == "" {
    ///     ..
    /// }
    ///
    /// if arr == [] {
    ///     ..
    /// }
    /// ```
    /// Use instead:
    /// ```ignore
    /// if s.is_empty() {
    ///     ..
    /// }
    ///
    /// if arr.is_empty() {
    ///     ..
    /// }
    /// ```
    pub COMPARISON_TO_EMPTY,
    style,
    "checking `x == \"\"` or `x == []` (or similar) when `.is_empty()` could be used instead"
}

declare_lint_pass!(LenZero => [LEN_ZERO, LEN_WITHOUT_IS_EMPTY, COMPARISON_TO_EMPTY]);

impl<'tcx> LateLintPass<'tcx> for LenZero {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        if item.span.from_expansion() {
            return;
        }

        if let ItemKind::Trait(_, _, _, _, trait_items) = item.kind {
            check_trait_items(cx, item, trait_items);
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx ImplItem<'_>) {
        if_chain! {
            if item.ident.name == sym::len;
            if let ImplItemKind::Fn(sig, _) = &item.kind;
            if sig.decl.implicit_self.has_implicit_self();
            if cx.access_levels.is_exported(item.def_id);
            if matches!(sig.decl.output, FnRetTy::Return(_));
            if let Some(imp) = get_parent_as_impl(cx.tcx, item.hir_id());
            if imp.of_trait.is_none();
            if let TyKind::Path(ty_path) = &imp.self_ty.kind;
            if let Some(ty_id) = cx.qpath_res(ty_path, imp.self_ty.hir_id).opt_def_id();
            if let Some(local_id) = ty_id.as_local();
            let ty_hir_id = cx.tcx.hir().local_def_id_to_hir_id(local_id);
            if !is_lint_allowed(cx, LEN_WITHOUT_IS_EMPTY, ty_hir_id);
            if let Some(output) = parse_len_output(cx, cx.tcx.fn_sig(item.def_id).skip_binder());
            then {
                let (name, kind) = match cx.tcx.hir().find(ty_hir_id) {
                    Some(Node::ForeignItem(x)) => (x.ident.name, "extern type"),
                    Some(Node::Item(x)) => match x.kind {
                        ItemKind::Struct(..) => (x.ident.name, "struct"),
                        ItemKind::Enum(..) => (x.ident.name, "enum"),
                        ItemKind::Union(..) => (x.ident.name, "union"),
                        _ => (x.ident.name, "type"),
                    }
                    _ => return,
                };
                check_for_is_empty(cx, sig.span, sig.decl.implicit_self, output, ty_id, name, kind)
            }
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if expr.span.from_expansion() {
            return;
        }

        if let ExprKind::Binary(Spanned { node: cmp, .. }, left, right) = expr.kind {
            match cmp {
                BinOpKind::Eq => {
                    check_cmp(cx, expr.span, left, right, "", 0); // len == 0
                    check_cmp(cx, expr.span, right, left, "", 0); // 0 == len
                },
                BinOpKind::Ne => {
                    check_cmp(cx, expr.span, left, right, "!", 0); // len != 0
                    check_cmp(cx, expr.span, right, left, "!", 0); // 0 != len
                },
                BinOpKind::Gt => {
                    check_cmp(cx, expr.span, left, right, "!", 0); // len > 0
                    check_cmp(cx, expr.span, right, left, "", 1); // 1 > len
                },
                BinOpKind::Lt => {
                    check_cmp(cx, expr.span, left, right, "", 1); // len < 1
                    check_cmp(cx, expr.span, right, left, "!", 0); // 0 < len
                },
                BinOpKind::Ge => check_cmp(cx, expr.span, left, right, "!", 1), // len >= 1
                BinOpKind::Le => check_cmp(cx, expr.span, right, left, "!", 1), // 1 <= len
                _ => (),
            }
        }
    }
}

fn check_trait_items(cx: &LateContext<'_>, visited_trait: &Item<'_>, trait_items: &[TraitItemRef]) {
    fn is_named_self(cx: &LateContext<'_>, item: &TraitItemRef, name: Symbol) -> bool {
        item.ident.name == name
            && if let AssocItemKind::Fn { has_self } = item.kind {
                has_self && { cx.tcx.fn_sig(item.id.def_id).inputs().skip_binder().len() == 1 }
            } else {
                false
            }
    }

    // fill the set with current and super traits
    fn fill_trait_set(traitt: DefId, set: &mut DefIdSet, cx: &LateContext<'_>) {
        if set.insert(traitt) {
            for supertrait in rustc_trait_selection::traits::supertrait_def_ids(cx.tcx, traitt) {
                fill_trait_set(supertrait, set, cx);
            }
        }
    }

    if cx.access_levels.is_exported(visited_trait.def_id) && trait_items.iter().any(|i| is_named_self(cx, i, sym::len))
    {
        let mut current_and_super_traits = DefIdSet::default();
        fill_trait_set(visited_trait.def_id.to_def_id(), &mut current_and_super_traits, cx);

        let is_empty_method_found = current_and_super_traits
            .iter()
            .flat_map(|&i| cx.tcx.associated_items(i).in_definition_order())
            .any(|i| {
                i.kind == ty::AssocKind::Fn
                    && i.fn_has_self_parameter
                    && i.ident.name == sym!(is_empty)
                    && cx.tcx.fn_sig(i.def_id).inputs().skip_binder().len() == 1
            });

        if !is_empty_method_found {
            span_lint(
                cx,
                LEN_WITHOUT_IS_EMPTY,
                visited_trait.span,
                &format!(
                    "trait `{}` has a `len` method but no (possibly inherited) `is_empty` method",
                    visited_trait.ident.name
                ),
            );
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum LenOutput<'tcx> {
    Integral,
    Option(DefId),
    Result(DefId, Ty<'tcx>),
}
fn parse_len_output(cx: &LateContext<'_>, sig: FnSig<'tcx>) -> Option<LenOutput<'tcx>> {
    match *sig.output().kind() {
        ty::Int(_) | ty::Uint(_) => Some(LenOutput::Integral),
        ty::Adt(adt, subs) if cx.tcx.is_diagnostic_item(sym::Option, adt.did) => {
            subs.type_at(0).is_integral().then(|| LenOutput::Option(adt.did))
        },
        ty::Adt(adt, subs) if cx.tcx.is_diagnostic_item(sym::Result, adt.did) => subs
            .type_at(0)
            .is_integral()
            .then(|| LenOutput::Result(adt.did, subs.type_at(1))),
        _ => None,
    }
}

impl LenOutput<'_> {
    fn matches_is_empty_output(self, ty: Ty<'_>) -> bool {
        match (self, ty.kind()) {
            (_, &ty::Bool) => true,
            (Self::Option(id), &ty::Adt(adt, subs)) if id == adt.did => subs.type_at(0).is_bool(),
            (Self::Result(id, err_ty), &ty::Adt(adt, subs)) if id == adt.did => {
                subs.type_at(0).is_bool() && TyS::same_type(subs.type_at(1), err_ty)
            },
            _ => false,
        }
    }

    fn expected_sig(self, self_kind: ImplicitSelfKind) -> String {
        let self_ref = match self_kind {
            ImplicitSelfKind::ImmRef => "&",
            ImplicitSelfKind::MutRef => "&mut ",
            _ => "",
        };
        match self {
            Self::Integral => format!("expected signature: `({}self) -> bool`", self_ref),
            Self::Option(_) => format!(
                "expected signature: `({}self) -> bool` or `({}self) -> Option<bool>",
                self_ref, self_ref
            ),
            Self::Result(..) => format!(
                "expected signature: `({}self) -> bool` or `({}self) -> Result<bool>",
                self_ref, self_ref
            ),
        }
    }
}

/// Checks if the given signature matches the expectations for `is_empty`
fn check_is_empty_sig(sig: FnSig<'_>, self_kind: ImplicitSelfKind, len_output: LenOutput<'_>) -> bool {
    match &**sig.inputs_and_output {
        [arg, res] if len_output.matches_is_empty_output(res) => {
            matches!(
                (arg.kind(), self_kind),
                (ty::Ref(_, _, Mutability::Not), ImplicitSelfKind::ImmRef)
                    | (ty::Ref(_, _, Mutability::Mut), ImplicitSelfKind::MutRef)
            ) || (!arg.is_ref() && matches!(self_kind, ImplicitSelfKind::Imm | ImplicitSelfKind::Mut))
        },
        _ => false,
    }
}

/// Checks if the given type has an `is_empty` method with the appropriate signature.
fn check_for_is_empty(
    cx: &LateContext<'_>,
    span: Span,
    self_kind: ImplicitSelfKind,
    output: LenOutput<'_>,
    impl_ty: DefId,
    item_name: Symbol,
    item_kind: &str,
) {
    let is_empty = Symbol::intern("is_empty");
    let is_empty = cx
        .tcx
        .inherent_impls(impl_ty)
        .iter()
        .flat_map(|&id| cx.tcx.associated_items(id).filter_by_name_unhygienic(is_empty))
        .find(|item| item.kind == AssocKind::Fn);

    let (msg, is_empty_span, self_kind) = match is_empty {
        None => (
            format!(
                "{} `{}` has a public `len` method, but no `is_empty` method",
                item_kind,
                item_name.as_str(),
            ),
            None,
            None,
        ),
        Some(is_empty) if !cx.access_levels.is_exported(is_empty.def_id.expect_local()) => (
            format!(
                "{} `{}` has a public `len` method, but a private `is_empty` method",
                item_kind,
                item_name.as_str(),
            ),
            Some(cx.tcx.def_span(is_empty.def_id)),
            None,
        ),
        Some(is_empty)
            if !(is_empty.fn_has_self_parameter
                && check_is_empty_sig(cx.tcx.fn_sig(is_empty.def_id).skip_binder(), self_kind, output)) =>
        {
            (
                format!(
                    "{} `{}` has a public `len` method, but the `is_empty` method has an unexpected signature",
                    item_kind,
                    item_name.as_str(),
                ),
                Some(cx.tcx.def_span(is_empty.def_id)),
                Some(self_kind),
            )
        },
        Some(_) => return,
    };

    span_lint_and_then(cx, LEN_WITHOUT_IS_EMPTY, span, &msg, |db| {
        if let Some(span) = is_empty_span {
            db.span_note(span, "`is_empty` defined here");
        }
        if let Some(self_kind) = self_kind {
            db.note(&output.expected_sig(self_kind));
        }
    });
}

fn check_cmp(cx: &LateContext<'_>, span: Span, method: &Expr<'_>, lit: &Expr<'_>, op: &str, compare_to: u32) {
    if let (&ExprKind::MethodCall(method_path, _, args, _), &ExprKind::Lit(ref lit)) = (&method.kind, &lit.kind) {
        // check if we are in an is_empty() method
        if let Some(name) = get_item_name(cx, method) {
            if name.as_str() == "is_empty" {
                return;
            }
        }

        check_len(cx, span, method_path.ident.name, args, &lit.node, op, compare_to);
    } else {
        check_empty_expr(cx, span, method, lit, op);
    }
}

fn check_len(
    cx: &LateContext<'_>,
    span: Span,
    method_name: Symbol,
    args: &[Expr<'_>],
    lit: &LitKind,
    op: &str,
    compare_to: u32,
) {
    if let LitKind::Int(lit, _) = *lit {
        // check if length is compared to the specified number
        if lit != u128::from(compare_to) {
            return;
        }

        if method_name == sym::len && args.len() == 1 && has_is_empty(cx, &args[0]) {
            let mut applicability = Applicability::MachineApplicable;
            span_lint_and_sugg(
                cx,
                LEN_ZERO,
                span,
                &format!("length comparison to {}", if compare_to == 0 { "zero" } else { "one" }),
                &format!("using `{}is_empty` is clearer and more explicit", op),
                format!(
                    "{}{}.is_empty()",
                    op,
                    snippet_with_applicability(cx, args[0].span, "_", &mut applicability)
                ),
                applicability,
            );
        }
    }
}

fn check_empty_expr(cx: &LateContext<'_>, span: Span, lit1: &Expr<'_>, lit2: &Expr<'_>, op: &str) {
    if (is_empty_array(lit2) || is_empty_string(lit2)) && has_is_empty(cx, lit1) {
        let mut applicability = Applicability::MachineApplicable;
        span_lint_and_sugg(
            cx,
            COMPARISON_TO_EMPTY,
            span,
            "comparison to empty slice",
            &format!("using `{}is_empty` is clearer and more explicit", op),
            format!(
                "{}{}.is_empty()",
                op,
                snippet_with_applicability(cx, lit1.span, "_", &mut applicability)
            ),
            applicability,
        );
    }
}

fn is_empty_string(expr: &Expr<'_>) -> bool {
    if let ExprKind::Lit(ref lit) = expr.kind {
        if let LitKind::Str(lit, _) = lit.node {
            let lit = lit.as_str();
            return lit == "";
        }
    }
    false
}

fn is_empty_array(expr: &Expr<'_>) -> bool {
    if let ExprKind::Array(arr) = expr.kind {
        return arr.is_empty();
    }
    false
}

/// Checks if this type has an `is_empty` method.
fn has_is_empty(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    /// Gets an `AssocItem` and return true if it matches `is_empty(self)`.
    fn is_is_empty(cx: &LateContext<'_>, item: &ty::AssocItem) -> bool {
        if item.kind == ty::AssocKind::Fn && item.ident.name.as_str() == "is_empty" {
            let sig = cx.tcx.fn_sig(item.def_id);
            let ty = sig.skip_binder();
            ty.inputs().len() == 1
        } else {
            false
        }
    }

    /// Checks the inherent impl's items for an `is_empty(self)` method.
    fn has_is_empty_impl(cx: &LateContext<'_>, id: DefId) -> bool {
        cx.tcx.inherent_impls(id).iter().any(|imp| {
            cx.tcx
                .associated_items(*imp)
                .in_definition_order()
                .any(|item| is_is_empty(cx, item))
        })
    }

    let ty = &cx.typeck_results().expr_ty(expr).peel_refs();
    match ty.kind() {
        ty::Dynamic(tt, ..) => tt.principal().map_or(false, |principal| {
            cx.tcx
                .associated_items(principal.def_id())
                .in_definition_order()
                .any(|item| is_is_empty(cx, item))
        }),
        ty::Projection(ref proj) => has_is_empty_impl(cx, proj.item_def_id),
        ty::Adt(id, _) => has_is_empty_impl(cx, id.did),
        ty::Array(..) | ty::Slice(..) | ty::Str => true,
        _ => false,
    }
}
