use rustc::lint::*;
use rustc::hir;
use rustc::ty;
use syntax_pos::Span;
use utils::{method_chain_args, match_def_path, span_lint_and_then, walk_ptrs_ty};
use utils::paths::{BEGIN_PANIC, BEGIN_PANIC_FMT, FROM_TRAIT, OPTION, RESULT, STRING};

/// **What it does:** Checks for impls of `From<&str>` and `From<String>` that contain `panic!()` or
/// `unwrap()`
///
/// **Why is this bad?** `FromStr` should be used if there's a possibility of failure.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// struct Foo(i32);
/// impl From<String> for Foo {
///     fn from(s: String) -> Self {
///         Foo(s.parse().unwrap())
///     }
/// }
/// ```
declare_lint! {
    pub IMPL_FROM_STR, Warn,
    "Warn on impls of `From<&str>` and `From<String>` that contain `panic!()` or `unwrap()`"
}

pub struct ImplFromStr;

impl LintPass for ImplFromStr {
    fn get_lints(&self) -> LintArray {
        lint_array!(IMPL_FROM_STR)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for ImplFromStr {
    fn check_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx hir::Item) {
        // check for `impl From<???> for ..`
        let impl_def_id = cx.tcx.hir.local_def_id(item.id);
        if_let_chain!{[
            let hir::ItemImpl(.., ref impl_items) = item.node,
            let Some(impl_trait_ref) = cx.tcx.impl_trait_ref(impl_def_id),
            match_def_path(cx.tcx, impl_trait_ref.def_id, &FROM_TRAIT),
        ], {
            // check if the type parameter is `str` or `String`
            let from_ty_param = impl_trait_ref.substs.type_at(1);
            let base_from_ty_param =
                walk_ptrs_ty(cx.tcx.normalize_associated_type(&from_ty_param));
            if base_from_ty_param.sty == ty::TyStr ||
                match_type(cx.tcx, base_from_ty_param, &STRING)
            {
                lint_impl_body(cx, item.span, impl_items);
            }
        }}
    }
}

fn lint_impl_body<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, impl_span: Span, impl_items: &hir::HirVec<hir::ImplItemRef>) {
    use rustc::hir::*;
    use rustc::hir::intravisit::{self, NestedVisitorMap, Visitor};

    struct FindPanicUnwrap<'a, 'tcx: 'a> {
        tcx: ty::TyCtxt<'a, 'tcx, 'tcx>,
        tables: &'tcx ty::TypeckTables<'tcx>,
        result: Vec<Span>,
    }

    impl<'a, 'tcx: 'a> Visitor<'tcx> for FindPanicUnwrap<'a, 'tcx> {
        fn visit_expr(&mut self, expr: &'tcx Expr) {
            // check for `begin_panic`
            if_let_chain!{[
                let ExprCall(ref func_expr, _) = expr.node,
                let ExprPath(QPath::Resolved(_, ref path)) = func_expr.node,
                match_def_path(self.tcx, path.def.def_id(), &BEGIN_PANIC) ||
                    match_def_path(self.tcx, path.def.def_id(), &BEGIN_PANIC_FMT),
            ], {
                self.result.push(expr.span);
            }}

            // check for `unwrap`
            if let Some(arglists) = method_chain_args(expr, &["unwrap"]) {
                let reciever_ty = walk_ptrs_ty(self.tables.expr_ty(&arglists[0][0]));
                if match_type(self.tcx, reciever_ty, &OPTION) ||
                    match_type(self.tcx, reciever_ty, &RESULT)
                {
                    self.result.push(expr.span);
                }
            }

            // and check sub-expressions
            intravisit::walk_expr(self, expr);
        }

        fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
            NestedVisitorMap::None
        }
    }

    for impl_item in impl_items {
        if_let_chain!{[
            impl_item.name == "from",
            let ImplItemKind::Method(_, body_id) =
                cx.tcx.hir.impl_item(impl_item.id).node,
        ], {
            // check the body for `begin_panic` or `unwrap`
            let body = cx.tcx.hir.body(body_id);
            let impl_item_def_id = cx.tcx.hir.local_def_id(impl_item.id.node_id);
            let mut fpu = FindPanicUnwrap {
                tcx: cx.tcx,
                tables: cx.tcx.typeck_tables_of(impl_item_def_id),
                result: Vec::new(),
            };
            fpu.visit_expr(&body.value);

            // if we've found one, lint
            if !fpu.result.is_empty() {
                span_lint_and_then(
                    cx,
                    IMPL_FROM_STR,
                    impl_span,
                    "consider implementing `FromStr` instead",
                    move |db| {
                        db.span_note(fpu.result, "potential failure(s)");
                    });
            }
        }}
    }
}

fn match_type(tcx: ty::TyCtxt, ty: ty::Ty, path: &[&str]) -> bool {
    match ty.sty {
        ty::TyAdt(adt, _) => match_def_path(tcx, adt.did, path),
        _ => false,
    }
}
