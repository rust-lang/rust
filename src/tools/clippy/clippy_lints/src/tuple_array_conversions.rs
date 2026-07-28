use arrayvec::ArrayVec;
use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::res::MaybeResPath;
use clippy_utils::visitors::{Visitable, for_each_expr};
use clippy_utils::{SpanlessEq, is_from_proc_macro};
use core::ops::ControlFlow::{Break, Continue};
use core::{iter, mem};
use rustc_ast::LitKind;
use rustc_ast::visit::{VisitorResult, try_visit, visit_opt, walk_list};
use rustc_data_structures::packed::Pu128;
use rustc_hir::intravisit::Visitor;
use rustc_hir::{Arm, Expr, ExprKind, HirId, ImplItemKind, ItemKind, Node, PatKind, Stmt, TraitFn, TraitItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self, TyCtxt};
use rustc_session::impl_lint_pass;
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for tuple<=>array conversions that are not done with `.into()`.
    ///
    /// ### Why is this bad?
    /// It may be unnecessary complexity. `.into()` works for converting tuples<=> arrays of up to
    /// 12 elements and conveys the intent more clearly, while also leaving less room for hard to
    /// spot bugs!
    ///
    /// ### Known issues
    /// The suggested code may hide potential asymmetry in some cases. See
    /// [#11085](https://github.com/rust-lang/rust-clippy/issues/11085) for more info.
    ///
    /// ### Example
    /// ```rust,ignore
    /// let t1 = &[(1, 2), (3, 4)];
    /// let v1: Vec<[u32; 2]> = t1.iter().map(|&(a, b)| [a, b]).collect();
    /// ```
    /// Use instead:
    /// ```rust,ignore
    /// let t1 = &[(1, 2), (3, 4)];
    /// let v1: Vec<[u32; 2]> = t1.iter().map(|&t| t.into()).collect();
    /// ```
    #[clippy::version = "1.72.0"]
    pub TUPLE_ARRAY_CONVERSIONS,
    nursery,
    "checks for tuple<=>array conversions that are not done with `.into()`"
}

impl_lint_pass!(TupleArrayConversions => [TUPLE_ARRAY_CONVERSIONS]);

pub struct TupleArrayConversions {
    msrv: Msrv,
}
impl TupleArrayConversions {
    pub fn new(conf: &'static Conf) -> Self {
        Self { msrv: conf.msrv }
    }
}

/// The maximum size of a tuple/array for which `From` is implemented.
const MAX_CVT_COUNT: usize = 12;

impl LateLintPass<'_> for TupleArrayConversions {
    #[expect(clippy::too_many_lines)]
    fn check_expr<'tcx>(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'tcx>) {
        match e.kind {
            ExprKind::Array([]) | ExprKind::Tup([]) => {},
            ExprKind::Array(es) | ExprKind::Tup(es) if es.len() > MAX_CVT_COUNT => {},

            // Create an array using every item in a tuple (e.g. `[x.0, x.1, x.2]`).
            ExprKind::Array([first, rest @ ..]) if let ExprKind::Field(first_base, first_field) = first.kind => {
                if first_field.name == sym::integer(0)
                    && let ctxt = e.span.ctxt()
                    && ctxt == first.span.ctxt()
                    && ctxt == first_field.span.ctxt()
                    // Check that the remaining elements are accesses to the expected field.
                    && let Some(bases) = rest.iter().enumerate().map(|(i, e)| {
                        if let ExprKind::Field(base, field) = e.kind
                            && field.name == sym::integer(i + 1)
                            && ctxt == e.span.ctxt()
                            && ctxt == field.span.ctxt()
                        {
                            Some(base)
                        } else {
                            None
                        }
                    }).collect::<Option<ArrayVec<_, MAX_CVT_COUNT>>>()
                    // Check that the source and destination types are the same and copyable.
                    && let ty::Tuple(src_tys) = *cx.typeck_results.expr_ty_adjusted(first_base).kind()
                    && src_tys.len() == rest.len() + 1
                    && let ty::Array(dst_ty, _) = *cx.typeck_results.expr_ty(e).kind()
                    && src_tys.iter().all(|ty| ty == dst_ty)
                    && cx.tcx.type_is_copy_modulo_regions(cx.typing_env(), dst_ty)
                    // Check that all accesses are to the same base last as that can be a complex check.
                    && let mut eq = SpanlessEq::new(cx).deny_side_effects()
                    && bases.iter().all(|e| eq.eq_expr(ctxt, first_base, e))
                    && self.msrv.meets(cx, msrvs::TUPLE_ARRAY_CONVERSIONS)
                    && !ctxt.in_external_macro(cx.tcx.sess.source_map())
                    && !is_from_proc_macro(cx, e)
                {
                    span_lint_and_help(
                        cx,
                        TUPLE_ARRAY_CONVERSIONS,
                        e.span,
                        "it looks like you're trying to convert a tuple to an array",
                        None,
                        "use `.into()` instead, or `<[T; N]>::from` if type annotations are needed",
                    );
                }
            },

            // Create a tuple using every item in an array (e.g. `(x[0], x[1], x[2])`).
            ExprKind::Tup([first, rest @ ..]) if let ExprKind::Index(first_base, first_idx, _) = first.kind => {
                if let ExprKind::Lit(idx_lit) = first_idx.kind
                    && let LitKind::Int(Pu128(0), _) = idx_lit.node
                    && let ctxt = e.span.ctxt()
                    && ctxt == first.span.ctxt()
                    && ctxt == first_idx.span.ctxt()
                    && ctxt == idx_lit.span.ctxt()
                    // Check that the remaining elements are array accesses with the expected index.
                    && let Some(bases) = rest.iter().enumerate().map(|(i, e)| {
                        if let ExprKind::Index(base, idx, _) = e.kind
                            && let ExprKind::Lit(idx_lit) = idx.kind
                            && let LitKind::Int(idx_num, _) = idx_lit.node
                            && idx_num == Pu128((i + 1) as u128)
                            && ctxt == e.span.ctxt()
                            && ctxt == idx_lit.span.ctxt()
                        {
                            Some(base)
                        } else {
                            None
                        }
                    }).collect::<Option<ArrayVec<_, MAX_CVT_COUNT>>>()
                    // Check that the source and destination types are the same and copyable.
                    && let ty::Array(src_ty, src_len) = *cx.typeck_results.expr_ty_adjusted(first_base).kind()
                    && src_len.try_to_target_usize(cx.tcx) == Some((rest.len() + 1) as u64)
                    && let ty::Tuple(dst_tys) = *cx.typeck_results.expr_ty(e).kind()
                    && dst_tys.iter().all(|ty| ty == src_ty)
                    && cx.tcx.type_is_copy_modulo_regions(cx.typing_env(), src_ty)
                    // Check that all accesses are to the same base last as that can be a complex check.
                    && let mut eq = SpanlessEq::new(cx).deny_side_effects()
                    && bases.iter().all(|e| eq.eq_expr(ctxt, first_base, e))
                    && self.msrv.meets(cx, msrvs::TUPLE_ARRAY_CONVERSIONS)
                    && !ctxt.in_external_macro(cx.tcx.sess.source_map())
                    && !is_from_proc_macro(cx, e)
                {
                    span_lint_and_help(
                        cx,
                        TUPLE_ARRAY_CONVERSIONS,
                        e.span,
                        "it looks like you're trying to convert an array to a tuple",
                        None,
                        "use `.into()` instead, or `<(T0, T1, ..., Tn)>::from` if type annotations are needed",
                    );
                }
            },

            // Destructure an array/tuple and create the other by using all the items.
            // e.g. `|(x, y, z)| [x, y, z]`
            ExprKind::Array([first, rest @ ..]) | ExprKind::Tup([first, rest @ ..])
                if let Some((first_id, first_ident)) = first.res_local_id_and_ident()
                    && let ctxt = e.span.ctxt()
                    && ctxt == first.span.ctxt()
                    && ctxt == first_ident.span.ctxt()
                    // Collect all the remaining local IDs involved.
                    // This is done first so we don't go through `TyCtxt` unless needed.
                    && let Some(mut ids) = rest
                        .iter()
                        .map(|e| {
                            e.res_local_id_and_ident()
                                .filter(|&(_, ident)| ctxt == e.span.ctxt() && ctxt == ident.span.ctxt())
                                .map(|(id, _)| id)
                        })
                        .collect::<Option<ArrayVec<_, MAX_CVT_COUNT>>>()
                    // Check that all locals used are from a single destructuring in the same order.
                    // The iterator will be used later to get the enclosing scope for the bindings.
                    && let mut parent_iter = cx.tcx.hir_parent_iter(first_id)
                    && let Some((id_parent, Node::Pat(id_parent_pat))) = parent_iter.next()
                    && let PatKind::Tuple([first_pat, rest_pats @ ..], _)
                    | PatKind::Slice([first_pat, rest_pats @ ..], ..) = id_parent_pat.kind
                    && first_id == first_pat.hir_id
                    && let PatKind::Binding(.., None) = first_pat.kind
                    && iter::zip(&ids, rest_pats)
                        .all(|(&id, pat)| id == pat.hir_id && matches!(pat.kind, PatKind::Binding(.., None)))
                    // Check that the source and destination types are the same.
                    && let Some(src_ty) = match *cx.typeck_results.node_type(id_parent).kind() {
                        ty::Array(src_ty, src_len)
                            if matches!(e.kind, ExprKind::Tup(_))
                                && src_len.try_to_target_usize(cx.tcx) == Some((rest.len() + 1) as u64) =>
                        {
                            Some(src_ty)
                        },
                        ty::Tuple(src_tys)
                            if  let [src_ty, ref rest_tys @ ..] = **src_tys
                                && rest.len() == rest_tys.len()
                                && matches!(e.kind, ExprKind::Array(_))
                                && rest_tys.iter().all(|&ty| src_ty == ty) =>
                        {
                            Some(src_ty)
                        },
                        _ => None,
                    }
                    && match *cx.typeck_results.expr_ty(e).kind() {
                        ty::Array(dst_ty, _) => dst_ty == src_ty,
                        ty::Tuple(dst_tys) => dst_tys.iter().all(|ty| src_ty == ty),
                        __ => false,
                    }
                    && ctxt == id_parent_pat.span.ctxt()
                    // Check that each binding is used at most once.
                    && let Some(use_scope) = find_binding_use_scope(cx.tcx, parent_iter)
                    && let () = ids.push(first_id)
                    && let mut ids_used = [false; MAX_CVT_COUNT]
                    && let None = for_each_expr(cx.tcx, use_scope, |e| {
                        if let Some(id) = e.res_local_id()
                            && let Some(i) = ids.iter().position(|&x| id == x)
                            && mem::replace(&mut ids_used[i], true)
                        {
                            Break(())
                        } else {
                            Continue(())
                        }
                    })
                    && !ctxt.in_external_macro(cx.tcx.sess.source_map())
                    && !is_from_proc_macro(cx, e) =>
            {
                let (msg, help) = if let ExprKind::Array(_) = e.kind {
                    (
                        "it looks like you're trying to convert a tuple to an array",
                        "use `.into()` instead, or `<[T; N]>::from` if type annotations are needed",
                    )
                } else {
                    (
                        "it looks like you're trying to convert an array to a tuple",
                        "use `.into()` instead, or `<(T0, T1, ..., Tn)>::from` if type annotations are needed",
                    )
                };
                span_lint_and_help(cx, TUPLE_ARRAY_CONVERSIONS, e.span, msg, None, help);
            },

            _ => {},
        }
    }
}

#[derive(Clone, Copy)]
enum BindingUseRange<'tcx> {
    Arm(&'tcx Arm<'tcx>),
    Param(&'tcx Expr<'tcx>),
    LetStmt(&'tcx [Stmt<'tcx>], Option<&'tcx Expr<'tcx>>),
    LetExpr(&'tcx Expr<'tcx>, &'tcx Expr<'tcx>),
}
impl<'tcx> Visitable<'tcx> for BindingUseRange<'tcx> {
    fn visit<V: Visitor<'tcx>>(self, visitor: &mut V) -> V::Result {
        match self {
            Self::Arm(a) => visitor.visit_arm(a),
            Self::Param(e) => visitor.visit_expr(e),
            Self::LetStmt(stmts, e) => {
                walk_list!(visitor, visit_stmt, stmts);
                visit_opt!(visitor, visit_expr, e);
                <V::Result as VisitorResult>::output()
            },
            Self::LetExpr(cond, then) => {
                try_visit!(visitor.visit_expr(cond));
                visitor.visit_expr(then)
            },
        }
    }
}

/// Given a parent iterator from a binding node, finds the parts of the HIR tree which can
/// use that binding.
fn find_binding_use_scope<'tcx>(
    tcx: TyCtxt<'tcx>,
    mut iter: impl Iterator<Item = (HirId, Node<'tcx>)>,
) -> Option<BindingUseRange<'tcx>> {
    loop {
        match iter.next() {
            Some((_, Node::Pat(p))) if !matches!(p.kind, PatKind::Or(_)) => {},
            Some((_, Node::PatField(_))) => {},
            Some((_, Node::Arm(a))) => break Some(BindingUseRange::Arm(a)),
            Some((_, Node::Param(_))) => {
                let body = match iter.next() {
                    Some((_, Node::Expr(e))) if let ExprKind::Closure(c) = e.kind => c.body,
                    Some((_, Node::Item(i))) if let ItemKind::Fn { body, .. } = i.kind => body,
                    Some((_, Node::TraitItem(i))) if let TraitItemKind::Fn(_, TraitFn::Provided(body)) = i.kind => body,
                    Some((_, Node::ImplItem(i))) if let ImplItemKind::Fn(_, body) = i.kind => body,
                    _ => break None,
                };
                break Some(BindingUseRange::Param(tcx.hir_body(body).value));
            },
            Some((_, Node::LetStmt(_)))
                if let Some((id, Node::Stmt(_))) = iter.next()
                    && let Some((_, Node::Block(b))) = iter.next()
                    && let mut stmts = b.stmts.iter()
                    && stmts.any(|s| s.hir_id == id) =>
            {
                break Some(BindingUseRange::LetStmt(stmts.as_slice(), b.expr));
            },
            Some((_, Node::Expr(e)))
                if let ExprKind::Let(_) = e.kind
                    && let Some((cond, then)) = iter.find_map(|(_, n)| match n {
                        Node::Expr(e) if let ExprKind::If(cond, then, _) = e.kind => Some((cond, then)),
                        _ => None,
                    }) =>
            {
                break Some(BindingUseRange::LetExpr(cond, then));
            },
            _ => break None,
        }
    }
}
