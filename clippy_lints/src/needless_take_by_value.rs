use rustc::hir::*;
use rustc::hir::intravisit::FnKind;
use rustc::hir::def_id::DefId;
use rustc::lint::*;
use rustc::ty;
use rustc::middle::expr_use_visitor as euv;
use rustc::middle::mem_categorization as mc;
use syntax::ast::NodeId;
use syntax_pos::Span;
use utils::{in_macro, is_self, is_copy, implements_trait, get_trait_def_id, snippet, span_lint_and_then, paths};
use std::collections::HashSet;

/// **What it does:** Checks for functions taking arguments by value, but only using them by
/// reference.
///
/// **Why is this bad?** In such cases, taking arguments by reference is more flexible and
/// can sometimes avoid unnecessary allocations.
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
    pub NEEDLESS_TAKE_BY_VALUE,
    Warn,
    "taking arguments by value, but only using them by reference"
}

pub struct NeedlessTakeByValue;

impl LintPass for NeedlessTakeByValue {
    fn get_lints(&self) -> LintArray {
        lint_array![NEEDLESS_TAKE_BY_VALUE]
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for NeedlessTakeByValue {
    fn check_fn(
        &mut self,
        cx: &LateContext<'a, 'tcx>,
        kind: FnKind<'tcx>,
        decl: &'tcx FnDecl,
        body: &'tcx Body,
        span: Span,
        node_id: NodeId
    ) {
        if in_macro(cx, span) {
            return;
        }

        if let FnKind::ItemFn(..) = kind {
        } else {
            return;
        }

        // These are usually took by value and only used by reference
        let fn_trait = cx.tcx.lang_items.fn_trait().expect("failed to find `Fn` trait");
        let asref_trait = get_trait_def_id(cx, &paths::ASREF_TRAIT).expect("failed to find `AsRef` trait");
        let borrow_trait = get_trait_def_id(cx, &paths::BORROW_TRAIT).expect("failed to find `Borrow` trait");

        // Collect moved variables from the function body
        let moved_vars = {
            let mut ctx = MovedVariablesCtxt::new(cx);
            let infcx = cx.tcx.borrowck_fake_infer_ctxt(body.id());
            {
                let mut v = euv::ExprUseVisitor::new(&mut ctx, &infcx);
                v.consume_body(body);
            }
            ctx.moved_vars
        };

        let fn_def_id = cx.tcx.hir.local_def_id(node_id);
        let param_env = ty::ParameterEnvironment::for_item(cx.tcx, node_id);
        let fn_sig = cx.tcx.item_type(fn_def_id).fn_sig();
        let fn_sig = cx.tcx.liberate_late_bound_regions(param_env.free_id_outlive, fn_sig);

        for ((input, ty), arg) in decl.inputs.iter().zip(fn_sig.inputs()).zip(&body.arguments) {
            if_let_chain! {[
                !is_self(arg),
                !ty.is_mutable_pointer(),
                !is_copy(cx, ty, node_id),
                !implements_trait(cx, ty, fn_trait, &[], Some(node_id)),
                !implements_trait(cx, ty, asref_trait, &[], Some(node_id)),
                !implements_trait(cx, ty, borrow_trait, &[], Some(node_id)),

                let PatKind::Binding(mode, defid, ..) = arg.pat.node,
                !moved_vars.contains(&defid),
            ], {
                // Note: `toplevel_ref_arg` warns if `BindByRef`
                let m = match mode {
                    BindingMode::BindByRef(m) | BindingMode::BindByValue(m) => m,
                };
                if m == Mutability::MutMutable {
                    continue;
                }

                span_lint_and_then(cx,
                          NEEDLESS_TAKE_BY_VALUE,
                          input.span,
                          "this function taking a value by value, but only using them by reference",
                          |db| {
                              db.span_suggestion(input.span,
                                                 "consider taking a reference instead",
                                                 format!("&{}", snippet(cx, input.span, "_")));
                          });
            }}
        }
    }
}

struct MovedVariablesCtxt<'a, 'tcx: 'a> {
    cx: &'a LateContext<'a, 'tcx>,
    moved_vars: HashSet<DefId>,
}

impl<'a, 'tcx: 'a> MovedVariablesCtxt<'a, 'tcx> {
    fn new(cx: &'a LateContext<'a, 'tcx>) -> Self {
        MovedVariablesCtxt {
            cx: cx,
            moved_vars: HashSet::new(),
        }
    }

    fn consume_common(
        &mut self,
        _consume_id: NodeId,
        _consume_span: Span,
        cmt: mc::cmt<'tcx>,
        mode: euv::ConsumeMode
    ) {
        if_let_chain! {[
            let euv::ConsumeMode::Move(_) = mode,
            let mc::Categorization::Local(vid) = cmt.cat,
        ], {
            if let Some(def_id) = self.cx.tcx.hir.opt_local_def_id(vid) {
                self.moved_vars.insert(def_id);
            }
        }}

    }
}

impl<'a, 'tcx: 'a> euv::Delegate<'tcx> for MovedVariablesCtxt<'a, 'tcx> {
    fn consume(&mut self, consume_id: NodeId, consume_span: Span, cmt: mc::cmt<'tcx>, mode: euv::ConsumeMode) {
        self.consume_common(consume_id, consume_span, cmt, mode);
    }

    fn matched_pat(&mut self, _matched_pat: &Pat, _cmt: mc::cmt, _mode: euv::MatchMode) {}

    fn consume_pat(&mut self, consume_pat: &Pat, cmt: mc::cmt<'tcx>, mode: euv::ConsumeMode) {
        self.consume_common(consume_pat.id, consume_pat.span, cmt, mode);
    }

    fn borrow(
        &mut self,
        _borrow_id: NodeId,
        _borrow_span: Span,
        _cmt: mc::cmt<'tcx>,
        _loan_region: &'tcx ty::Region,
        _bk: ty::BorrowKind,
        _loan_cause: euv::LoanCause
    ) {
    }

    fn mutate(
        &mut self,
        _assignment_id: NodeId,
        _assignment_span: Span,
        _assignee_cmt: mc::cmt<'tcx>,
        _mode: euv::MutateMode
    ) {
    }

    fn decl_without_init(&mut self, _id: NodeId, _span: Span) {}
}
