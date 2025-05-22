use std::borrow::Cow;
use std::collections::BTreeMap;

use rustc_errors::{Applicability, Diag};
use rustc_hir::intravisit::{Visitor, VisitorExt, walk_body, walk_expr, walk_ty};
use rustc_hir::{self as hir, AmbigArg, Body, Expr, ExprKind, GenericArg, Item, ItemKind, QPath, TyKind};
use rustc_hir_analysis::lower_ty;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::nested_filter;
use rustc_middle::ty::{Ty, TypeckResults};
use rustc_session::declare_lint_pass;
use rustc_span::Span;

use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::{IntoSpan, SpanRangeExt, snippet};
use clippy_utils::sym;
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
    /// ```no_run
    /// # use std::collections::HashMap;
    /// # use std::hash::{Hash, BuildHasher};
    /// # trait Serialize {};
    /// impl<K: Hash + Eq, V> Serialize for HashMap<K, V> { }
    ///
    /// pub fn foo(map: &mut HashMap<i32, i32>) { }
    /// ```
    /// could be rewritten as
    /// ```no_run
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
    #[expect(clippy::too_many_lines)]
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        fn suggestion(
            cx: &LateContext<'_>,
            diag: &mut Diag<'_, ()>,
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

            let mut suggestions = vec![
                (
                    generics_suggestion_span,
                    format!(
                        "<{generics_snip}{}S: ::std::hash::BuildHasher{}>",
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
            ];
            suggestions.extend(vis.suggestions);

            diag.multipart_suggestion(
                "add a type parameter for `BuildHasher`",
                suggestions,
                Applicability::MaybeIncorrect,
            );
        }

        if !cx.effective_visibilities.is_exported(item.owner_id.def_id) {
            return;
        }

        match item.kind {
            ItemKind::Impl(impl_) => {
                let mut vis = ImplicitHasherTypeVisitor::new(cx);
                vis.visit_ty_unambig(impl_.self_ty);

                for target in &vis.found {
                    if !item.span.eq_ctxt(target.span()) {
                        return;
                    }

                    let generics_suggestion_span = impl_.generics.span.substitute_dummy({
                        let range = (item.span.lo()..target.span().lo()).map_range(cx, |_, src, range| {
                            Some(src.get(range.clone())?.find("impl")? + 4..range.end)
                        });
                        if let Some(range) = range {
                            range.with_ctxt(item.span.ctxt())
                        } else {
                            return;
                        }
                    });

                    let mut ctr_vis = ImplicitHasherConstructorVisitor::new(cx, target);
                    for item in impl_.items.iter().map(|item| cx.tcx.hir_impl_item(item.id)) {
                        ctr_vis.visit_impl_item(item);
                    }

                    span_lint_and_then(
                        cx,
                        IMPLICIT_HASHER,
                        target.span(),
                        format!(
                            "impl for `{}` should be generalized over different hashers",
                            target.type_name()
                        ),
                        move |diag| {
                            suggestion(cx, diag, impl_.generics.span, generics_suggestion_span, target, ctr_vis);
                        },
                    );
                }
            },
            ItemKind::Fn {
                ref sig,
                generics,
                body: body_id,
                ..
            } => {
                let body = cx.tcx.hir_body(body_id);

                for ty in sig.decl.inputs {
                    let mut vis = ImplicitHasherTypeVisitor::new(cx);
                    vis.visit_ty_unambig(ty);

                    for target in &vis.found {
                        if generics.span.from_expansion() {
                            continue;
                        }
                        let generics_suggestion_span = generics.span.substitute_dummy({
                            let range =
                                (item.span.lo()..body.params[0].pat.span.lo()).map_range(cx, |_, src, range| {
                                    let (pre, post) = src.get(range.clone())?.split_once("fn")?;
                                    let pos = post.find('(')? + pre.len() + 2;
                                    Some(pos..pos)
                                });
                            if let Some(range) = range {
                                range.with_ctxt(item.span.ctxt())
                            } else {
                                return;
                            }
                        });

                        let mut ctr_vis = ImplicitHasherConstructorVisitor::new(cx, target);
                        ctr_vis.visit_body(body);

                        span_lint_and_then(
                            cx,
                            IMPLICIT_HASHER,
                            target.span(),
                            format!(
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
    fn new(cx: &LateContext<'tcx>, hir_ty: &hir::Ty<'tcx>) -> Option<Self> {
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

            let ty = lower_ty(cx.tcx, hir_ty);

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
            ImplicitHasherType::HashMap(.., ref k, ref v) => format!("{k}, {v}"),
            ImplicitHasherType::HashSet(.., ref t) => format!("{t}"),
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

impl<'tcx> Visitor<'tcx> for ImplicitHasherTypeVisitor<'_, 'tcx> {
    fn visit_ty(&mut self, t: &'tcx hir::Ty<'_, AmbigArg>) {
        if let Some(target) = ImplicitHasherType::new(self.cx, t.as_unambig_ty()) {
            self.found.push(target);
        }

        walk_ty(self, t);
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

impl<'tcx> Visitor<'tcx> for ImplicitHasherConstructorVisitor<'_, '_, 'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;

    fn visit_body(&mut self, body: &Body<'tcx>) {
        let old_maybe_typeck_results = self.maybe_typeck_results.replace(self.cx.tcx.typeck_body(body.id()));
        walk_body(self, body);
        self.maybe_typeck_results = old_maybe_typeck_results;
    }

    fn visit_expr(&mut self, e: &'tcx Expr<'_>) {
        if let ExprKind::Call(fun, args) = e.kind
            && let ExprKind::Path(QPath::TypeRelative(ty, method)) = fun.kind
            && matches!(method.ident.name, sym::new | sym::with_capacity)
            && let TyKind::Path(QPath::Resolved(None, ty_path)) = ty.kind
            && let Some(ty_did) = ty_path.res.opt_def_id()
        {
            if self.target.ty() != self.maybe_typeck_results.unwrap().expr_ty(e) {
                return;
            }

            match (self.cx.tcx.get_diagnostic_name(ty_did), method.ident.name) {
                (Some(sym::HashMap), sym::new) => {
                    self.suggestions.insert(e.span, "HashMap::default()".to_string());
                },
                (Some(sym::HashMap), sym::with_capacity) => {
                    self.suggestions.insert(
                        e.span,
                        format!(
                            "HashMap::with_capacity_and_hasher({}, Default::default())",
                            snippet(self.cx, args[0].span, "capacity"),
                        ),
                    );
                },
                (Some(sym::HashSet), sym::new) => {
                    self.suggestions.insert(e.span, "HashSet::default()".to_string());
                },
                (Some(sym::HashSet), sym::with_capacity) => {
                    self.suggestions.insert(
                        e.span,
                        format!(
                            "HashSet::with_capacity_and_hasher({}, Default::default())",
                            snippet(self.cx, args[0].span, "capacity"),
                        ),
                    );
                },
                _ => {},
            }
        }

        walk_expr(self, e);
    }

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.cx.tcx
    }
}
