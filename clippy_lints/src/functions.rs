use rustc::hir::intravisit;
use rustc::hir;
use rustc::ty;
use rustc::lint::*;
use std::collections::HashSet;
use syntax::ast;
use syntax::codemap::Span;
use utils::{span_lint, type_is_unsafe_function};

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
/// ```rust
/// fn foo(x: u32, y: u32, name: &str, c: Color, w: f32, h: f32, a: f32, b: f32) { .. }
/// ```
declare_lint! {
    pub TOO_MANY_ARGUMENTS,
    Warn,
    "functions with too many arguments"
}

/// **What it does:** Check for public functions that dereferences raw pointer arguments but are
/// not marked unsafe.
///
/// **Why is this bad?** The function should probably be marked `unsafe`, since for an arbitrary
/// raw pointer, there is no way of telling for sure if it is valid.
///
/// **Known problems:**
///
/// * It does not check functions recursively so if the pointer is passed to a private non-
/// `unsafe` function which does the dereferencing, the lint won't trigger.
/// * It only checks for arguments whose type are raw pointers, not raw pointers got from an
/// argument in some other way (`fn foo(bar: &[*const u8])` or `some_argument.get_raw_ptr()`).
///
/// **Example:**
///
/// ```rust
/// pub fn foo(x: *const u8) { println!("{}", unsafe { *x }); }
/// ```
declare_lint! {
    pub NOT_UNSAFE_PTR_ARG_DEREF,
    Warn,
    "public functions dereferencing raw pointer arguments but not marked `unsafe`"
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
        lint_array!(TOO_MANY_ARGUMENTS, NOT_UNSAFE_PTR_ARG_DEREF)
    }
}

impl LateLintPass for Functions {
    fn check_fn(&mut self, cx: &LateContext, kind: intravisit::FnKind, decl: &hir::FnDecl, block: &hir::Block, span: Span, nodeid: ast::NodeId) {
        use rustc::hir::map::Node::*;

        let is_impl = if let Some(NodeItem(ref item)) = cx.tcx.map.find(cx.tcx.map.get_parent_node(nodeid)) {
            matches!(item.node, hir::ItemImpl(_, _, _, Some(_), _, _) | hir::ItemDefaultImpl(..))
        } else {
            false
        };

        let unsafety = match kind {
            hir::intravisit::FnKind::ItemFn(_, _, unsafety, _, _, _, _) => unsafety,
            hir::intravisit::FnKind::Method(_, sig, _, _) => sig.unsafety,
            hir::intravisit::FnKind::Closure(_) => return,
        };

        // don't warn for implementations, it's not their fault
        if !is_impl {
            self.check_arg_number(cx, decl, span);
        }

        self.check_raw_ptr(cx, unsafety, decl, block, span, nodeid);
    }

    fn check_trait_item(&mut self, cx: &LateContext, item: &hir::TraitItem) {
        if let hir::MethodTraitItem(ref sig, ref block) = item.node {
            self.check_arg_number(cx, &sig.decl, item.span);

            if let Some(ref block) = *block {
                self.check_raw_ptr(cx, sig.unsafety, &sig.decl, block, item.span, item.id);
            }
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

    fn check_raw_ptr(&self, cx: &LateContext, unsafety: hir::Unsafety, decl: &hir::FnDecl, block: &hir::Block, span: Span, nodeid: ast::NodeId) {
        if unsafety == hir::Unsafety::Normal && cx.access_levels.is_exported(nodeid) {
            let raw_ptrs = decl.inputs.iter().filter_map(|arg| raw_ptr_arg(cx, arg)).collect::<HashSet<_>>();

            if !raw_ptrs.is_empty() {
                let mut v = DerefVisitor {
                    cx: cx,
                    ptrs: raw_ptrs,
                };

                hir::intravisit::walk_block(&mut v, block);
            }
        }
    }
}

fn raw_ptr_arg(cx: &LateContext, arg: &hir::Arg) -> Option<hir::def_id::DefId> {
    if let (&hir::PatKind::Binding(_, _, _), &hir::TyPtr(_)) = (&arg.pat.node, &arg.ty.node) {
        cx.tcx.def_map.borrow().get(&arg.pat.id).map(hir::def::PathResolution::def_id)
    } else {
        None
    }
}

struct DerefVisitor<'a, 'tcx: 'a> {
    cx: &'a LateContext<'a, 'tcx>,
    ptrs: HashSet<hir::def_id::DefId>,
}

impl<'a, 'tcx, 'v> hir::intravisit::Visitor<'v> for DerefVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, expr: &'v hir::Expr) {
        let ptr = match expr.node {
            hir::ExprUnary(hir::UnDeref, ref ptr) => Some(ptr),
            hir::ExprMethodCall(_, _, ref args) => {
                let method_call = ty::MethodCall::expr(expr.id);
                let base_type = self.cx.tcx.tables.borrow().method_map[&method_call].ty;

                if type_is_unsafe_function(base_type) {
                    Some(&args[0])
                } else {
                    None
                }
            }
            _ => None,
        };

        if let Some(ptr) = ptr {
            if let Some(def) = self.cx.tcx.def_map.borrow().get(&ptr.id) {
                if self.ptrs.contains(&def.def_id()) {
                    span_lint(self.cx,
                              NOT_UNSAFE_PTR_ARG_DEREF,
                              ptr.span,
                              "this public function dereferences a raw pointer but is not marked `unsafe`");
                }
            }
        }

        hir::intravisit::walk_expr(self, expr);
    }
}
