use if_chain::if_chain;
use rustc::hir::map::Map;
use rustc_hir::def::Res;
use rustc_hir::intravisit::{walk_path, NestedVisitorMap, Visitor};
use rustc_hir::{AssocItemKind, HirId, ImplItem, ImplItemKind, ImplItemRef, ItemKind, Path};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

use crate::utils::span_help_and_lint;

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

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for UnusedSelf {
    fn check_impl_item(&mut self, cx: &LateContext<'a, 'tcx>, impl_item: &ImplItem<'_>) {
        if impl_item.span.from_expansion() {
            return;
        }
        let parent = cx.tcx.hir().get_parent_item(impl_item.hir_id);
        let item = cx.tcx.hir().expect_item(parent);
        if let ItemKind::Impl {
            of_trait: None,
            items: impl_item_refs,
            ..
        } = item.kind
        {
            for impl_item_ref in impl_item_refs {
                if_chain! {
                    if let ImplItemRef {
                        kind: AssocItemKind::Method { has_self: true },
                        ..
                    } = impl_item_ref;
                    if let ImplItemKind::Method(_, body_id) = &impl_item.kind;
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
                            span_help_and_lint(
                                cx,
                                UNUSED_SELF,
                                self_param.span,
                                "unused `self` argument",
                                "consider refactoring to a associated function",
                            );
                            return;
                        }
                    }
                }
            }
        };
    }
}

struct UnusedSelfVisitor<'a, 'tcx> {
    cx: &'a LateContext<'a, 'tcx>,
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

    fn nested_visit_map(&mut self) -> NestedVisitorMap<'_, Self::Map> {
        NestedVisitorMap::OnlyBodies(&self.cx.tcx.hir())
    }
}
