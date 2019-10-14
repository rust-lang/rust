use if_chain::if_chain;
use rustc::hir::def::Res;
use rustc::hir::intravisit::{walk_path, NestedVisitorMap, Visitor};
use rustc::hir::{AssocItemKind, HirId, ImplItemKind, ImplItemRef, Item, ItemKind, Path};
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};

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
    fn check_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &Item) {
        if item.span.from_expansion() {
            return;
        }
        if let ItemKind::Impl(_, _, _, _, None, _, ref impl_item_refs) = item.kind {
            for impl_item_ref in impl_item_refs {
                if_chain! {
                    if let ImplItemRef {
                        kind: AssocItemKind::Method { has_self: true },
                        ..
                    } = impl_item_ref;
                    let impl_item = cx.tcx.hir().impl_item(impl_item_ref.id);
                    if let ImplItemKind::Method(_, body_id) = &impl_item.kind;
                    then {
                        let body = cx.tcx.hir().body(*body_id);
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
                            )
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
    fn visit_path(&mut self, path: &'tcx Path, _id: HirId) {
        if self.uses_self {
            // This function already uses `self`
            return;
        }
        if let Res::Local(hir_id) = &path.res {
            self.uses_self = self.self_hir_id == hir_id
        }
        walk_path(self, path);
    }

    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::OnlyBodies(&self.cx.tcx.hir())
    }
}
