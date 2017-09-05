use rustc::hir::intravisit;
use rustc::hir;
use rustc::lint::*;
use rustc::ty;
use std::collections::HashSet;
use syntax::ast;
use syntax::abi::Abi;
use syntax::codemap::Span;
use utils::{iter_input_pats, span_lint, type_is_unsafe_function};

/// **What it does:** Checks for functions with too many parameters.
///
/// **Why is this bad?** Functions with lots of parameters are considered bad
/// style and reduce readability (“what does the 5th parameter mean?”). Consider
/// grouping some parameters into a new type.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// fn foo(x: u32, y: u32, name: &str, c: Color, w: f32, h: f32, a: f32, b:
/// f32) { .. }
/// ```
declare_lint! {
    pub TOO_MANY_ARGUMENTS,
    Warn,
    "functions with too many arguments"
}

/// **What it does:** Checks for public functions that dereferences raw pointer
/// arguments but are not marked unsafe.
///
/// **Why is this bad?** The function should probably be marked `unsafe`, since
/// for an arbitrary raw pointer, there is no way of telling for sure if it is
/// valid.
///
/// **Known problems:**
///
/// * It does not check functions recursively so if the pointer is passed to a
/// private non-`unsafe` function which does the dereferencing, the lint won't
/// trigger.
/// * It only checks for arguments whose type are raw pointers, not raw pointers
/// got from an argument in some other way (`fn foo(bar: &[*const u8])` or
/// `some_argument.get_raw_ptr()`).
///
/// **Example:**
/// ```rust
/// pub fn foo(x: *const u8) { println!("{}", unsafe { *x }); }
/// ```
declare_lint! {
    pub NOT_UNSAFE_PTR_ARG_DEREF,
    Warn,
    "public functions dereferencing raw pointer arguments but not marked `unsafe`"
}

#[derive(Copy, Clone)]
pub struct Functions {
    threshold: u64,
}

impl Functions {
    pub fn new(threshold: u64) -> Self {
        Self {
            threshold: threshold,
        }
    }
}

impl LintPass for Functions {
    fn get_lints(&self) -> LintArray {
        lint_array!(TOO_MANY_ARGUMENTS, NOT_UNSAFE_PTR_ARG_DEREF)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Functions {
    fn check_fn(
        &mut self,
        cx: &LateContext<'a, 'tcx>,
        kind: intravisit::FnKind<'tcx>,
        decl: &'tcx hir::FnDecl,
        body: &'tcx hir::Body,
        span: Span,
        nodeid: ast::NodeId,
    ) {
        use rustc::hir::map::Node::*;

        let is_impl = if let Some(NodeItem(item)) = cx.tcx.hir.find(cx.tcx.hir.get_parent_node(nodeid)) {
            matches!(item.node, hir::ItemImpl(_, _, _, _, Some(_), _, _) | hir::ItemDefaultImpl(..))
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
            // don't lint extern functions decls, it's not their fault either
            match kind {
                hir::intravisit::FnKind::Method(_, &hir::MethodSig { abi: Abi::Rust, .. }, _, _) |
                hir::intravisit::FnKind::ItemFn(_, _, _, _, Abi::Rust, _, _) => self.check_arg_number(cx, decl, span),
                _ => {},
            }
        }

        self.check_raw_ptr(cx, unsafety, decl, body, nodeid);
    }

    fn check_trait_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx hir::TraitItem) {
        if let hir::TraitItemKind::Method(ref sig, ref eid) = item.node {
            // don't lint extern functions decls, it's not their fault
            if sig.abi == Abi::Rust {
                self.check_arg_number(cx, &sig.decl, item.span);
            }

            if let hir::TraitMethod::Provided(eid) = *eid {
                let body = cx.tcx.hir.body(eid);
                self.check_raw_ptr(cx, sig.unsafety, &sig.decl, body, item.id);
            }
        }
    }
}

impl<'a, 'tcx> Functions {
    fn check_arg_number(&self, cx: &LateContext, decl: &hir::FnDecl, span: Span) {
        let args = decl.inputs.len() as u64;
        if args > self.threshold {
            span_lint(
                cx,
                TOO_MANY_ARGUMENTS,
                span,
                &format!("this function has too many arguments ({}/{})", args, self.threshold),
            );
        }
    }

    fn check_raw_ptr(
        &self,
        cx: &LateContext<'a, 'tcx>,
        unsafety: hir::Unsafety,
        decl: &'tcx hir::FnDecl,
        body: &'tcx hir::Body,
        nodeid: ast::NodeId,
    ) {
        let expr = &body.value;
        if unsafety == hir::Unsafety::Normal && cx.access_levels.is_exported(nodeid) {
            let raw_ptrs = iter_input_pats(decl, body)
                .zip(decl.inputs.iter())
                .filter_map(|(arg, ty)| raw_ptr_arg(arg, ty))
                .collect::<HashSet<_>>();

            if !raw_ptrs.is_empty() {
                let tables = cx.tcx.body_tables(body.id());
                let mut v = DerefVisitor {
                    cx: cx,
                    ptrs: raw_ptrs,
                    tables,
                };

                hir::intravisit::walk_expr(&mut v, expr);
            }
        }
    }
}

fn raw_ptr_arg(arg: &hir::Arg, ty: &hir::Ty) -> Option<hir::def_id::DefId> {
    if let (&hir::PatKind::Binding(_, def_id, _, _), &hir::TyPtr(_)) = (&arg.pat.node, &ty.node) {
        Some(def_id)
    } else {
        None
    }
}

struct DerefVisitor<'a, 'tcx: 'a> {
    cx: &'a LateContext<'a, 'tcx>,
    ptrs: HashSet<hir::def_id::DefId>,
    tables: &'a ty::TypeckTables<'tcx>,
}

impl<'a, 'tcx> hir::intravisit::Visitor<'tcx> for DerefVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx hir::Expr) {
        match expr.node {
            hir::ExprCall(ref f, ref args) => {
                let ty = self.tables.expr_ty(f);

                if type_is_unsafe_function(self.cx, ty) {
                    for arg in args {
                        self.check_arg(arg);
                    }
                }
            },
            hir::ExprMethodCall(_, _, ref args) => {
                let def_id = self.tables.type_dependent_defs()[expr.hir_id].def_id();
                let base_type = self.cx.tcx.type_of(def_id);

                if type_is_unsafe_function(self.cx, base_type) {
                    for arg in args {
                        self.check_arg(arg);
                    }
                }
            },
            hir::ExprUnary(hir::UnDeref, ref ptr) => self.check_arg(ptr),
            _ => (),
        }

        hir::intravisit::walk_expr(self, expr);
    }
    fn nested_visit_map<'this>(&'this mut self) -> intravisit::NestedVisitorMap<'this, 'tcx> {
        intravisit::NestedVisitorMap::None
    }
}

impl<'a, 'tcx: 'a> DerefVisitor<'a, 'tcx> {
    fn check_arg(&self, ptr: &hir::Expr) {
        if let hir::ExprPath(ref qpath) = ptr.node {
            let def = self.cx.tables.qpath_def(qpath, ptr.hir_id);
            if self.ptrs.contains(&def.def_id()) {
                span_lint(
                    self.cx,
                    NOT_UNSAFE_PTR_ARG_DEREF,
                    ptr.span,
                    "this public function dereferences a raw pointer but is not marked `unsafe`",
                );
            }
        }
    }
}
