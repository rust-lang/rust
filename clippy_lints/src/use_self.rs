use rustc::lint::{LintArray, LateLintPass, LateContext, LintPass};
use rustc::hir::*;
use rustc::hir::intravisit::{Visitor, walk_path, NestedVisitorMap};
use utils::{span_lint_and_then, in_macro};
use syntax::ast::NodeId;
use syntax_pos::symbol::keywords::SelfType;

/// **What it does:** Checks for unnecessary repetition of structure name when a
/// replacement with `Self` is applicable.
///
/// **Why is this bad?** Unnecessary repetition. Mixed use of `Self` and struct
/// name
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
    "Unnecessary structure name repetition whereas `Self` is applicable"
}

#[derive(Copy, Clone, Default)]
pub struct UseSelf;

impl LintPass for UseSelf {
    fn get_lints(&self) -> LintArray {
        lint_array!(USE_SELF)
    }
}

const SEGMENTS_MSG: &str = "segments should be composed of at least 1 element";

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for UseSelf {
    fn check_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx Item) {
        if in_macro(item.span) {
            return;
        }
        if_let_chain!([
            let ItemImpl(.., ref item_type, ref refs) = item.node,
            let Ty_::TyPath(QPath::Resolved(_, ref item_path)) = item_type.node,
        ], {
            let parameters = &item_path.segments.last().expect(SEGMENTS_MSG).parameters;
            if !parameters.parenthesized && parameters.lifetimes.len() == 0 {
                let visitor = &mut UseSelfVisitor {
                    item_path: item_path,
                    cx: cx,
                };
                for impl_item_ref in refs {
                    visitor.visit_impl_item(cx.tcx.hir.impl_item(impl_item_ref.id));
                }
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
        if self.item_path.def == path.def && path.segments.last().expect(SEGMENTS_MSG).name != SelfType.name() {
            span_lint_and_then(self.cx, USE_SELF, path.span, "unnecessary structure name repetition", |db| {
                db.span_suggestion(path.span, "use the applicable keyword", "Self".to_owned());
            });
        }

        walk_path(self, path);
    }

    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::OnlyBodies(&self.cx.tcx.hir)
    }
}
