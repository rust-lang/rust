use std::borrow::Cow;
use std::collections::BTreeMap;

use rustc_errors::DiagnosticBuilder;
use rustc_hir as hir;
use rustc_hir::intravisit::{walk_body, walk_expr, walk_inf, walk_ty, NestedVisitorMap, Visitor};
use rustc_hir::{Body, Expr, ExprKind, GenericArg, Item, ItemKind, QPath, TyKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::hir::map::Map;
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::{Ty, TyS, TypeckResults};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;
use rustc_span::symbol::sym;
use rustc_typeck::hir_ty_to_ty;

use if_chain::if_chain;

use clippy_utils::diagnostics::{multispan_sugg, span_lint_and_then};
use clippy_utils::differing_macro_contexts;
use clippy_utils::source::{snippet, snippet_opt};
use clippy_utils::ty::is_type_diagnostic_item;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for public `impl` or `fn` missing generalization
    /// over different hashers and implicitly defaulting to the default hashing
    /// algorithm (`SipHash`).
    ///
    /// ### Why is this bad?
    /// `HashMap` or `HashSet` with custom hashers cannot be
    /// used with them.
    ///
    /// ### Known problems
    /// Suggestions for replacing constructors can contain
    /// false-positives. Also applying suggestions can require modification of other
    /// pieces of code, possibly including external crates.
    ///
    /// ### Example
    /// ```rust
    /// # use std::collections::HashMap;
    /// # use std::hash::{Hash, BuildHasher};
    /// # trait Serialize {};
    /// impl<K: Hash + Eq, V> Serialize for HashMap<K, V> { }
    ///
    /// pub fn foo(map: &mut HashMap<i32, i32>) { }
    /// ```
    /// could be rewritten as
    /// ```rust
    /// # use std::collections::HashMap;
    /// # use std::hash::{Hash, BuildHasher};
    /// # trait Serialize {};
    /// impl<K: Hash + Eq, V, S: BuildHasher> Serialize for HashMap<K, V, S> { }
    ///
    /// pub fn foo<S: BuildHasher>(map: &mut HashMap<i32, i32, S>) { }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub IMPLICIT_HASHER,
    pedantic,
    "missing generalization over different hashers"
}

declare_lint_pass!(ImplicitHasher => [IMPLICIT_HASHER]);

impl<'tcx> LateLintPass<'tcx> for ImplicitHasher {
    #[allow(clippy::cast_possible_truncation, clippy::too_many_lines)]
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        use rustc_span::BytePos;

        fn suggestion<'tcx>(
            cx: &LateContext<'tcx>,
            diag: &mut DiagnosticBuilder<'_>,
            generics_span: Span,
            generics_suggestion_span: Span,
            target: &ImplicitHasherType<'_>,
            vis: ImplicitHasherConstructorVisitor<'_, '_, '_>,
        ) {
            let generics_snip = snippet(cx, generics_span, "");
            // trim `<` `>`
            let generics_snip = if generics_snip.is_empty() {
                ""
            } else {
                &generics_snip[1..generics_snip.len() - 1]
            };

            multispan_sugg(
                diag,
                "consider adding a type parameter",
                vec![
                    (
                        generics_suggestion_span,
                        format!(
                            "<{}{}S: ::std::hash::BuildHasher{}>",
                            generics_snip,
                            if generics_snip.is_empty() { "" } else { ", " },
                            if vis.suggestions.is_empty() {
                                ""
                            } else {
                                // request users to add `Default` bound so that generic constructors can be used
                                " + Default"
                            },
                        ),
                    ),
                    (
                        target.span(),
                        format!("{}<{}, S>", target.type_name(), target.type_arguments(),),
                    ),
                ],
            );

            if !vis.suggestions.is_empty() {
                multispan_sugg(diag, "...and use generic constructor", vis.suggestions);
            }
        }

        if !cx.access_levels.is_exported(item.def_id) {
            return;
        }

        match item.kind {
            ItemKind::Impl(ref impl_) => {
                let mut vis = ImplicitHasherTypeVisitor::new(cx);
                vis.visit_ty(impl_.self_ty);

                for target in &vis.found {
                    if differing_macro_contexts(item.span, target.span()) {
                        return;
                    }

                    let generics_suggestion_span = impl_.generics.span.substitute_dummy({
                        let pos = snippet_opt(cx, item.span.until(target.span()))
                            .and_then(|snip| Some(item.span.lo() + BytePos(snip.find("impl")? as u32 + 4)));
                        if let Some(pos) = pos {
                            Span::new(pos, pos, item.span.ctxt(), item.span.parent())
                        } else {
                            return;
                        }
                    });

                    let mut ctr_vis = ImplicitHasherConstructorVisitor::new(cx, target);
                    for item in impl_.items.iter().map(|item| cx.tcx.hir().impl_item(item.id)) {
                        ctr_vis.visit_impl_item(item);
                    }

                    span_lint_and_then(
                        cx,
                        IMPLICIT_HASHER,
                        target.span(),
                        &format!(
                            "impl for `{}` should be generalized over different hashers",
                            target.type_name()
                        ),
                        move |diag| {
                            suggestion(cx, diag, impl_.generics.span, generics_suggestion_span, target, ctr_vis);
                        },
                    );
                }
            },
            ItemKind::Fn(ref sig, ref generics, body_id) => {
                let body = cx.tcx.hir().body(body_id);

                for ty in sig.decl.inputs {
                    let mut vis = ImplicitHasherTypeVisitor::new(cx);
                    vis.visit_ty(ty);

                    for target in &vis.found {
                        if in_external_macro(cx.sess(), generics.span) {
                            continue;
                        }
                        let generics_suggestion_span = generics.span.substitute_dummy({
                            let pos = snippet_opt(
                                cx,
                                Span::new(
                                    item.span.lo(),
                                    body.params[0].pat.span.lo(),
                                    item.span.ctxt(),
                                    item.span.parent(),
                                ),
                            )
                            .and_then(|snip| {
                                let i = snip.find("fn")?;
                                Some(item.span.lo() + BytePos((i + (&snip[i..]).find('(')?) as u32))
                            })
                            .expect("failed to create span for type parameters");
                            Span::new(pos, pos, item.span.ctxt(), item.span.parent())
                        });

                        let mut ctr_vis = ImplicitHasherConstructorVisitor::new(cx, target);
                        ctr_vis.visit_body(body);

                        span_lint_and_then(
                            cx,
                            IMPLICIT_HASHER,
                            target.span(),
                            &format!(
                                "parameter of type `{}` should be generalized over different hashers",
                                target.type_name()
                            ),
                            move |diag| {
                                suggestion(cx, diag, generics.span, generics_suggestion_span, target, ctr_vis);
                            },
                        );
                    }
                }
            },
            _ => {},
        }
    }
}

enum ImplicitHasherType<'tcx> {
    HashMap(Span, Ty<'tcx>, Cow<'static, str>, Cow<'static, str>),
    HashSet(Span, Ty<'tcx>, Cow<'static, str>),
}

impl<'tcx> ImplicitHasherType<'tcx> {
    /// Checks that `ty` is a target type without a `BuildHasher`.
    fn new(cx: &LateContext<'tcx>, hir_ty: &hir::Ty<'_>) -> Option<Self> {
        if let TyKind::Path(QPath::Resolved(None, path)) = hir_ty.kind {
            let params: Vec<_> = path
                .segments
                .last()
                .as_ref()?
                .args
                .as_ref()?
                .args
                .iter()
                .filter_map(|arg| match arg {
                    GenericArg::Type(ty) => Some(ty),
                    _ => None,
                })
                .collect();
            let params_len = params.len();

            let ty = hir_ty_to_ty(cx.tcx, hir_ty);

            if is_type_diagnostic_item(cx, ty, sym::HashMap) && params_len == 2 {
                Some(ImplicitHasherType::HashMap(
                    hir_ty.span,
                    ty,
                    snippet(cx, params[0].span, "K"),
                    snippet(cx, params[1].span, "V"),
                ))
            } else if is_type_diagnostic_item(cx, ty, sym::HashSet) && params_len == 1 {
                Some(ImplicitHasherType::HashSet(
                    hir_ty.span,
                    ty,
                    snippet(cx, params[0].span, "T"),
                ))
            } else {
                None
            }
        } else {
            None
        }
    }

    fn type_name(&self) -> &'static str {
        match *self {
            ImplicitHasherType::HashMap(..) => "HashMap",
            ImplicitHasherType::HashSet(..) => "HashSet",
        }
    }

    fn type_arguments(&self) -> String {
        match *self {
            ImplicitHasherType::HashMap(.., ref k, ref v) => format!("{}, {}", k, v),
            ImplicitHasherType::HashSet(.., ref t) => format!("{}", t),
        }
    }

    fn ty(&self) -> Ty<'tcx> {
        match *self {
            ImplicitHasherType::HashMap(_, ty, ..) | ImplicitHasherType::HashSet(_, ty, ..) => ty,
        }
    }

    fn span(&self) -> Span {
        match *self {
            ImplicitHasherType::HashMap(span, ..) | ImplicitHasherType::HashSet(span, ..) => span,
        }
    }
}

struct ImplicitHasherTypeVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    found: Vec<ImplicitHasherType<'tcx>>,
}

impl<'a, 'tcx> ImplicitHasherTypeVisitor<'a, 'tcx> {
    fn new(cx: &'a LateContext<'tcx>) -> Self {
        Self { cx, found: vec![] }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for ImplicitHasherTypeVisitor<'a, 'tcx> {
    type Map = Map<'tcx>;

    fn visit_ty(&mut self, t: &'tcx hir::Ty<'_>) {
        if let Some(target) = ImplicitHasherType::new(self.cx, t) {
            self.found.push(target);
        }

        walk_ty(self, t);
    }

    fn visit_infer(&mut self, inf: &'tcx hir::InferArg) {
        if let Some(target) = ImplicitHasherType::new(self.cx, &inf.to_ty()) {
            self.found.push(target);
        }

        walk_inf(self, inf);
    }

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }
}

/// Looks for default-hasher-dependent constructors like `HashMap::new`.
struct ImplicitHasherConstructorVisitor<'a, 'b, 'tcx> {
    cx: &'a LateContext<'tcx>,
    maybe_typeck_results: Option<&'tcx TypeckResults<'tcx>>,
    target: &'b ImplicitHasherType<'tcx>,
    suggestions: BTreeMap<Span, String>,
}

impl<'a, 'b, 'tcx> ImplicitHasherConstructorVisitor<'a, 'b, 'tcx> {
    fn new(cx: &'a LateContext<'tcx>, target: &'b ImplicitHasherType<'tcx>) -> Self {
        Self {
            cx,
            maybe_typeck_results: cx.maybe_typeck_results(),
            target,
            suggestions: BTreeMap::new(),
        }
    }
}

impl<'a, 'b, 'tcx> Visitor<'tcx> for ImplicitHasherConstructorVisitor<'a, 'b, 'tcx> {
    type Map = Map<'tcx>;

    fn visit_body(&mut self, body: &'tcx Body<'_>) {
        let old_maybe_typeck_results = self.maybe_typeck_results.replace(self.cx.tcx.typeck_body(body.id()));
        walk_body(self, body);
        self.maybe_typeck_results = old_maybe_typeck_results;
    }

    fn visit_expr(&mut self, e: &'tcx Expr<'_>) {
        if_chain! {
            if let ExprKind::Call(fun, args) = e.kind;
            if let ExprKind::Path(QPath::TypeRelative(ty, method)) = fun.kind;
            if let TyKind::Path(QPath::Resolved(None, ty_path)) = ty.kind;
            if let Some(ty_did) = ty_path.res.opt_def_id();
            then {
                if !TyS::same_type(self.target.ty(), self.maybe_typeck_results.unwrap().expr_ty(e)) {
                    return;
                }

                if self.cx.tcx.is_diagnostic_item(sym::HashMap, ty_did) {
                    if method.ident.name == sym::new {
                        self.suggestions
                            .insert(e.span, "HashMap::default()".to_string());
                    } else if method.ident.name == sym!(with_capacity) {
                        self.suggestions.insert(
                            e.span,
                            format!(
                                "HashMap::with_capacity_and_hasher({}, Default::default())",
                                snippet(self.cx, args[0].span, "capacity"),
                            ),
                        );
                    }
                } else if self.cx.tcx.is_diagnostic_item(sym::HashSet, ty_did) {
                    if method.ident.name == sym::new {
                        self.suggestions
                            .insert(e.span, "HashSet::default()".to_string());
                    } else if method.ident.name == sym!(with_capacity) {
                        self.suggestions.insert(
                            e.span,
                            format!(
                                "HashSet::with_capacity_and_hasher({}, Default::default())",
                                snippet(self.cx, args[0].span, "capacity"),
                            ),
                        );
                    }
                }
            }
        }

        walk_expr(self, e);
    }

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::OnlyBodies(self.cx.tcx.hir())
    }
}
