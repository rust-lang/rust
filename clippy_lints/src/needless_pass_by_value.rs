use rustc::hir::*;
use rustc::hir::intravisit::FnKind;
use rustc::hir::def_id::DefId;
use rustc::lint::*;
use rustc::ty::{self, TypeFoldable};
use rustc::traits;
use rustc::middle::expr_use_visitor as euv;
use rustc::middle::mem_categorization as mc;
use syntax::ast::NodeId;
use syntax_pos::Span;
use syntax::errors::DiagnosticBuilder;
use utils::{get_trait_def_id, implements_trait, in_macro, is_copy, is_self, match_type, multispan_sugg, paths,
            snippet, span_lint_and_then};
use std::collections::{HashMap, HashSet};

/// **What it does:** Checks for functions taking arguments by value, but not
/// consuming them in its
/// body.
///
/// **Why is this bad?** Taking arguments by reference is more flexible and can
/// sometimes avoid
/// unnecessary allocations.
///
/// **Known problems:** Hopefully none.
///
/// **Example:**
/// ```rust
/// fn foo(v: Vec<i32>) {
///     assert_eq!(v.len(), 42);
/// }
/// ```
declare_lint! {
    pub NEEDLESS_PASS_BY_VALUE,
    Warn,
    "functions taking arguments by value, but not consuming them in its body"
}

pub struct NeedlessPassByValue;

impl LintPass for NeedlessPassByValue {
    fn get_lints(&self) -> LintArray {
        lint_array![NEEDLESS_PASS_BY_VALUE]
    }
}

macro_rules! need {
    ($e: expr) => { if let Some(x) = $e { x } else { return; } };
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for NeedlessPassByValue {
    fn check_fn(
        &mut self,
        cx: &LateContext<'a, 'tcx>,
        kind: FnKind<'tcx>,
        decl: &'tcx FnDecl,
        body: &'tcx Body,
        span: Span,
        node_id: NodeId,
    ) {
        if in_macro(span) {
            return;
        }

        match kind {
            FnKind::ItemFn(.., attrs) => for a in attrs {
                if_let_chain!{[
                    a.meta_item_list().is_some(),
                    let Some(name) = a.name(),
                    name == "proc_macro_derive",
                ], {
                    return;
                }}
            },
            _ => return,
        }

        // Allows these to be passed by value.
        let fn_trait = need!(cx.tcx.lang_items().fn_trait());
        let asref_trait = need!(get_trait_def_id(cx, &paths::ASREF_TRAIT));
        let borrow_trait = need!(get_trait_def_id(cx, &paths::BORROW_TRAIT));

        let fn_def_id = cx.tcx.hir.local_def_id(node_id);

        let preds: Vec<ty::Predicate> = {
            traits::elaborate_predicates(cx.tcx, cx.param_env.caller_bounds.to_vec())
                .filter(|p| !p.is_global())
                .collect()
        };

        // Collect moved variables and spans which will need dereferencings from the
        // function body.
        let MovedVariablesCtxt {
            moved_vars,
            spans_need_deref,
            ..
        } = {
            let mut ctx = MovedVariablesCtxt::new(cx);
            let region_scope_tree = &cx.tcx.region_scope_tree(fn_def_id);
            euv::ExprUseVisitor::new(&mut ctx, cx.tcx, cx.param_env, region_scope_tree, cx.tables).consume_body(body);
            ctx
        };

        let fn_sig = cx.tcx.fn_sig(fn_def_id);
        let fn_sig = cx.tcx.erase_late_bound_regions(&fn_sig);

        for ((input, &ty), arg) in decl.inputs.iter().zip(fn_sig.inputs()).zip(&body.arguments) {
            // Determines whether `ty` implements `Borrow<U>` (U != ty) specifically.
            // This is needed due to the `Borrow<T> for T` blanket impl.
            let implements_borrow_trait = preds
                .iter()
                .filter_map(|pred| if let ty::Predicate::Trait(ref poly_trait_ref) = *pred {
                    Some(poly_trait_ref.skip_binder())
                } else {
                    None
                })
                .filter(|tpred| tpred.def_id() == borrow_trait && tpred.self_ty() == ty)
                .any(|tpred| {
                    tpred
                        .input_types()
                        .nth(1)
                        .expect("Borrow trait must have an parameter") != ty
                });

            if_let_chain! {[
                !is_self(arg),
                !ty.is_mutable_pointer(),
                !is_copy(cx, ty),
                !implements_trait(cx, ty, fn_trait, &[]),
                !implements_trait(cx, ty, asref_trait, &[]),
                !implements_borrow_trait,

                let PatKind::Binding(mode, defid, ..) = arg.pat.node,
                !moved_vars.contains(&defid),
            ], {
                // Note: `toplevel_ref_arg` warns if `BindByRef`
                if mode == BindingAnnotation::Mutable || mode == BindingAnnotation::RefMut {
                    continue;
                }

                // Suggestion logic
                let sugg = |db: &mut DiagnosticBuilder| {
                    let deref_span = spans_need_deref.get(&defid);
                    if_let_chain! {[
                        match_type(cx, ty, &paths::VEC),
                        let TyPath(QPath::Resolved(_, ref path)) = input.node,
                        let Some(elem_ty) = path.segments.iter()
                            .find(|seg| seg.name == "Vec")
                            .map(|ps| &ps.parameters.types[0]),
                    ], {
                        let slice_ty = format!("&[{}]", snippet(cx, elem_ty.span, "_"));
                        db.span_suggestion(input.span,
                                        "consider changing the type to",
                                        slice_ty);
                        assert!(deref_span.is_none());
                        return; // `Vec` and `String` cannot be destructured - no need for `*` suggestion
                    }}

                    if match_type(cx, ty, &paths::STRING) {
                        db.span_suggestion(input.span,
                                           "consider changing the type to",
                                           "&str".to_string());
                        assert!(deref_span.is_none());
                        return;
                    }

                    let mut spans = vec![(input.span, format!("&{}", snippet(cx, input.span, "_")))];

                    // Suggests adding `*` to dereference the added reference.
                    if let Some(deref_span) = deref_span {
                        spans.extend(deref_span.iter().cloned()
                                     .map(|span| (span, format!("*{}", snippet(cx, span, "<expr>")))));
                        spans.sort_by_key(|&(span, _)| span);
                    }
                    multispan_sugg(db, "consider taking a reference instead".to_string(), spans);
                };

                span_lint_and_then(cx,
                          NEEDLESS_PASS_BY_VALUE,
                          input.span,
                          "this argument is passed by value, but not consumed in the function body",
                          sugg);
            }}
        }
    }
}

struct MovedVariablesCtxt<'a, 'tcx: 'a> {
    cx: &'a LateContext<'a, 'tcx>,
    moved_vars: HashSet<DefId>,
    /// Spans which need to be prefixed with `*` for dereferencing the
    /// suggested additional
    /// reference.
    spans_need_deref: HashMap<DefId, HashSet<Span>>,
}

impl<'a, 'tcx> MovedVariablesCtxt<'a, 'tcx> {
    fn new(cx: &'a LateContext<'a, 'tcx>) -> Self {
        Self {
            cx: cx,
            moved_vars: HashSet::new(),
            spans_need_deref: HashMap::new(),
        }
    }

    fn move_common(&mut self, _consume_id: NodeId, _span: Span, cmt: mc::cmt<'tcx>) {
        let cmt = unwrap_downcast_or_interior(cmt);

        if_let_chain! {[
            let mc::Categorization::Local(vid) = cmt.cat,
            let Some(def_id) = self.cx.tcx.hir.opt_local_def_id(vid),
        ], {
                self.moved_vars.insert(def_id);
        }}
    }

    fn non_moving_pat(&mut self, matched_pat: &Pat, cmt: mc::cmt<'tcx>) {
        let cmt = unwrap_downcast_or_interior(cmt);

        if_let_chain! {[
            let mc::Categorization::Local(vid) = cmt.cat,
            let Some(def_id) = self.cx.tcx.hir.opt_local_def_id(vid),
        ], {
            let mut id = matched_pat.id;
            loop {
                let parent = self.cx.tcx.hir.get_parent_node(id);
                if id == parent {
                    // no parent
                    return;
                }
                id = parent;

                if let Some(node) = self.cx.tcx.hir.find(id) {
                    match node {
                        map::Node::NodeExpr(e) => {
                            // `match` and `if let`
                            if let ExprMatch(ref c, ..) = e.node {
                                self.spans_need_deref
                                    .entry(def_id)
                                    .or_insert_with(HashSet::new)
                                    .insert(c.span);
                            }
                        }

                        map::Node::NodeStmt(s) => {
                            // `let <pat> = x;`
                            if_let_chain! {[
                                let StmtDecl(ref decl, _) = s.node,
                                let DeclLocal(ref local) = decl.node,
                            ], {
                                self.spans_need_deref
                                    .entry(def_id)
                                    .or_insert_with(HashSet::new)
                                    .insert(local.init
                                        .as_ref()
                                        .map(|e| e.span)
                                        .expect("`let` stmt without init aren't caught by match_pat"));
                            }}
                        }

                        _ => {}
                    }
                }
            }
        }}
    }
}

impl<'a, 'tcx> euv::Delegate<'tcx> for MovedVariablesCtxt<'a, 'tcx> {
    fn consume(&mut self, consume_id: NodeId, consume_span: Span, cmt: mc::cmt<'tcx>, mode: euv::ConsumeMode) {
        if let euv::ConsumeMode::Move(_) = mode {
            self.move_common(consume_id, consume_span, cmt);
        }
    }

    fn matched_pat(&mut self, matched_pat: &Pat, cmt: mc::cmt<'tcx>, mode: euv::MatchMode) {
        if let euv::MatchMode::MovingMatch = mode {
            self.move_common(matched_pat.id, matched_pat.span, cmt);
        } else {
            self.non_moving_pat(matched_pat, cmt);
        }
    }

    fn consume_pat(&mut self, consume_pat: &Pat, cmt: mc::cmt<'tcx>, mode: euv::ConsumeMode) {
        if let euv::ConsumeMode::Move(_) = mode {
            self.move_common(consume_pat.id, consume_pat.span, cmt);
        }
    }

    fn borrow(&mut self, _: NodeId, _: Span, _: mc::cmt<'tcx>, _: ty::Region, _: ty::BorrowKind, _: euv::LoanCause) {}

    fn mutate(&mut self, _: NodeId, _: Span, _: mc::cmt<'tcx>, _: euv::MutateMode) {}

    fn decl_without_init(&mut self, _: NodeId, _: Span) {}
}


fn unwrap_downcast_or_interior(mut cmt: mc::cmt) -> mc::cmt {
    loop {
        match cmt.cat.clone() {
            mc::Categorization::Downcast(c, _) | mc::Categorization::Interior(c, _) => {
                cmt = c;
            },
            _ => return cmt,
        }
    }
}
