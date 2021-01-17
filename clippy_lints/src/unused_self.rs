use if_chain::if_chain;
use rustc_hir::def::Res;
use rustc_hir::intravisit::{walk_path, NestedVisitorMap, Visitor};
use rustc_hir::{HirId, Impl, ImplItem, ImplItemKind, ItemKind, Path};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::map::Map;
use rustc_session::{declare_lint_pass, declare_tool_lint};

use crate::utils::span_lint_and_help;

declare_clippy_lint! {
    /// **What it does:** Checks methods that contain a `self` argument but don't use it
    ///
    /// **Why is this bad?** It may be clearer to define the method as an associated function instead
    /// of an instance method if it doesn't require `self`.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust,ignore
    /// struct A;
    /// impl A {
    ///     fn method(&self) {}
    /// }
    /// ```
    ///
    /// Could be written:
    ///
    /// ```rust,ignore
    /// struct A;
    /// impl A {
    ///     fn method() {}
    /// }
    /// ```
    pub UNUSED_SELF,
    pedantic,
    "methods that contain a `self` argument but don't use it"
}

declare_lint_pass!(UnusedSelf => [UNUSED_SELF]);

impl<'tcx> LateLintPass<'tcx> for UnusedSelf {
    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, impl_item: &ImplItem<'_>) {
        if impl_item.span.from_expansion() {
            return;
        }
        let parent = cx.tcx.hir().get_parent_item(impl_item.hir_id);
        let parent_item = cx.tcx.hir().expect_item(parent);
        let def_id = cx.tcx.hir().local_def_id(impl_item.hir_id);
        let assoc_item = cx.tcx.associated_item(def_id);
        if_chain! {
            if let ItemKind::Impl(Impl { of_trait: None, .. }) = parent_item.kind;
            if assoc_item.fn_has_self_parameter;
            if let ImplItemKind::Fn(.., body_id) = &impl_item.kind;
            let body = cx.tcx.hir().body(*body_id);
            if !body.params.is_empty();
            then {
                let self_param = &body.params[0];
                let self_hir_id = self_param.pat.hir_id;
                let mut visitor = UnusedSelfVisitor {
                    cx,
                    uses_self: false,
                    self_hir_id: &self_hir_id,
                };
                visitor.visit_body(body);
                if !visitor.uses_self {
                    span_lint_and_help(
                        cx,
                        UNUSED_SELF,
                        self_param.span,
                        "unused `self` argument",
                        None,
                        "consider refactoring to a associated function",
                    );
                    return;
                }
            }
        }
    }
}

struct UnusedSelfVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    uses_self: bool,
    self_hir_id: &'a HirId,
}

impl<'a, 'tcx> Visitor<'tcx> for UnusedSelfVisitor<'a, 'tcx> {
    type Map = Map<'tcx>;

    fn visit_path(&mut self, path: &'tcx Path<'_>, _id: HirId) {
        if self.uses_self {
            // This function already uses `self`
            return;
        }
        if let Res::Local(hir_id) = &path.res {
            self.uses_self = self.self_hir_id == hir_id
        }
        walk_path(self, path);
    }

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::OnlyBodies(self.cx.tcx.hir())
    }
}
