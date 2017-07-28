use rustc::lint::{LintArray, LateLintPass, LateContext, LintPass};
use rustc::hir::*;
use rustc::hir::intravisit::{Visitor, walk_path, NestedVisitorMap};
use utils::span_lint;
use syntax::ast::NodeId;

/// **What it does:** Checks for unnecessary repetition of structure name when a
/// replacement with `Self` is applicable.
///
/// **Why is this bad?** Unnecessary repetition. Mixed use of `Self` and struct name
/// feels inconsistent.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// struct Foo {}
/// impl Foo {
///     fn new() -> Foo {
///         Foo {}
///     }
/// }
/// ```
/// could be
/// ```
/// struct Foo {}
/// impl Foo {
///     fn new() -> Self {
///         Self {}
///     }
/// }
/// ```
declare_lint! {
    pub USE_SELF,
    Allow,
    "Repetitive struct name usage whereas `Self` is applicable"
}

#[derive(Copy, Clone, Default)]
pub struct UseSelf;

impl LintPass for UseSelf {
    fn get_lints(&self) -> LintArray {
        lint_array!(USE_SELF)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for UseSelf {
    fn check_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx Item) {
        if_let_chain!([
            let ItemImpl(.., ref item_type, ref refs) = item.node,
            let Ty_::TyPath(QPath::Resolved(_, ref item_path)) = item_type.node,
        ], {
            let visitor = &mut UseSelfVisitor {
                item_path: item_path,
                cx: cx,
            };
            for impl_item_ref in refs {
                visitor.visit_impl_item(cx.tcx.hir.impl_item(impl_item_ref.id));
            }
        })
    }
}

struct UseSelfVisitor<'a, 'tcx: 'a> {
    item_path: &'a Path,
    cx: &'a LateContext<'a, 'tcx>,
}

impl<'a, 'tcx> Visitor<'tcx> for UseSelfVisitor<'a, 'tcx> {
    fn visit_path(&mut self, path: &'tcx Path, _id: NodeId) {
        if self.item_path.def == path.def &&
           path.segments
            .last()
            .expect("segments should be composed of at least 1 elemnt")
            .name
            .as_str() != "Self" {
            span_lint(self.cx,
                      USE_SELF,
                      path.span,
                      "repetitive struct name usage. Use `Self` instead.");
        }

        walk_path(self, path);
    }

    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::OnlyBodies(&self.cx.tcx.hir)
    }
}
