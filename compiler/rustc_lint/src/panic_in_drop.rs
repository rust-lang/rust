use crate::context::LintContext;
use crate::lints::PanicInDropAbortDiag;
use crate::{LateContext, LateLintPass};
use rustc_data_structures::fx::FxHashSet;
use rustc_hir::Node::Item;
use rustc_hir::{ImplItemKind, ItemKind, LangItem};
use rustc_middle::mir::{self, visit::Visitor};
use rustc_middle::ty;
use rustc_session::config::OptLevel;
use rustc_span::def_id::DefId;
use rustc_span::sym;
use rustc_target::spec::PanicStrategy;

declare_lint! {
    pub PANIC_IN_DROP,
    Warn,
    "detected panic inside Drop::drop call stack"
}

declare_lint_pass!(PanicInDrop => [PANIC_IN_DROP]);

impl<'tcx> LateLintPass<'tcx> for PanicInDrop {
    fn check_impl_item(
        &mut self,
        cx: &LateContext<'tcx>,
        impl_item: &'tcx rustc_hir::ImplItem<'tcx>,
    ) {
        if cx.tcx.sess.opts.unstable_opts.panic_in_drop != PanicStrategy::Abort
            || cx.tcx.sess.opts.optimize == OptLevel::No
            || cx.tcx.sess.opts.debug_assertions
        {
            return;
        }
        let hir_id = cx.tcx.hir().local_def_id_to_hir_id(impl_item.owner_id.def_id);
        let parent_impl = cx.tcx.hir().get_parent_item(hir_id);
        let trait_ref = if let Item(item) = cx.tcx.hir().get_by_def_id(parent_impl.def_id)
            && parent_impl != rustc_hir::CRATE_OWNER_ID
            && let ItemKind::Impl(impl_) = &item.kind
        {
            impl_.of_trait.as_ref().and_then(|t| t.trait_def_id())
        } else {
            None
        };

        if let ImplItemKind::Fn(..) = impl_item.kind
            &&  cx.tcx.lang_items().drop_trait() == trait_ref
            && impl_item.ident.name == sym::drop
        {
            let mir = cx.tcx.optimized_mir(impl_item.owner_id.def_id);
            let mut visitor = PanicFnVisitor::new(&cx);
            visitor.visit_body(mir);
        }
    }
}

struct PanicFnVisitor<'tcx, 'a> {
    cx: &'a LateContext<'tcx>,
}

impl<'a, 'tcx> PanicFnVisitor<'tcx, 'a> {
    fn new(cx: &'a LateContext<'tcx>) -> Self {
        PanicFnVisitor { cx }
    }
}

impl<'tcx, 'a> Visitor<'tcx> for PanicFnVisitor<'tcx, 'a> {
    fn visit_operand(&mut self, operand: &mir::Operand<'tcx>, location: mir::Location) {
        use mir::Operand::*;
        match operand {
            Constant(cst) => {
                if let ty::FnDef(def_id, _) = cst.literal.ty().kind() {
                    if self.cx.tcx.lang_items().get(LangItem::Panic) == Some(*def_id) {
                        self.cx.emit_spanned_lint(PANIC_IN_DROP, cst.span, PanicInDropAbortDiag {});
                    }
                }
            }
            _ => {}
        };
        self.super_operand(operand, location);
    }
}
