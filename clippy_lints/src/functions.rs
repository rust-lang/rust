use rustc::lint::*;
use rustc::hir;
use rustc::hir::intravisit;
use syntax::ast;
use syntax::codemap::Span;
use utils::span_lint;

/// **What it does:** Check for functions with too many parameters.
///
/// **Why is this bad?** Functions with lots of parameters are considered bad style and reduce
/// readability (“what does the 5th parameter mean?”). Consider grouping some parameters into a
/// new type.
///
/// **Known problems:** None.
///
/// **Example:**
///
/// ```
/// fn foo(x: u32, y: u32, name: &str, c: Color, w: f32, h: f32, a: f32, b: f32) { .. }
/// ```
declare_lint! {
    pub TOO_MANY_ARGUMENTS,
    Warn,
    "functions with too many arguments"
}

#[derive(Copy,Clone)]
pub struct Functions {
    threshold: u64,
}

impl Functions {
    pub fn new(threshold: u64) -> Functions {
        Functions { threshold: threshold }
    }
}

impl LintPass for Functions {
    fn get_lints(&self) -> LintArray {
        lint_array!(TOO_MANY_ARGUMENTS)
    }
}

impl LateLintPass for Functions {
    fn check_fn(&mut self, cx: &LateContext, _: intravisit::FnKind, decl: &hir::FnDecl, _: &hir::Block, span: Span,
                nodeid: ast::NodeId) {
        use rustc::hir::map::Node::*;

        if let Some(NodeItem(ref item)) = cx.tcx.map.find(cx.tcx.map.get_parent_node(nodeid)) {
            match item.node {
                hir::ItemImpl(_, _, _, Some(_), _, _) |
                hir::ItemDefaultImpl(..) => return,
                _ => (),
            }
        }

        self.check_arg_number(cx, decl, span);
    }

    fn check_trait_item(&mut self, cx: &LateContext, item: &hir::TraitItem) {
        if let hir::MethodTraitItem(ref sig, _) = item.node {
            self.check_arg_number(cx, &sig.decl, item.span);
        }
    }
}

impl Functions {
    fn check_arg_number(&self, cx: &LateContext, decl: &hir::FnDecl, span: Span) {
        let args = decl.inputs.len() as u64;
        if args > self.threshold {
            span_lint(cx,
                      TOO_MANY_ARGUMENTS,
                      span,
                      &format!("this function has too many arguments ({}/{})", args, self.threshold));
        }
    }
}
