use std::convert::TryFrom;

use crate::utils::{iter_input_pats, snippet, snippet_opt, span_lint, type_is_unsafe_function};
use matches::matches;
use rustc::hir;
use rustc::hir::def::Res;
use rustc::hir::intravisit;
use rustc::lint::{in_external_macro, LateContext, LateLintPass, LintArray, LintContext, LintPass};
use rustc::ty;
use rustc::{declare_tool_lint, impl_lint_pass};
use rustc_data_structures::fx::FxHashSet;
use rustc_target::spec::abi::Abi;
use syntax::source_map::{BytePos, Span};

declare_clippy_lint! {
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
    /// fn foo(x: u32, y: u32, name: &str, c: Color, w: f32, h: f32, a: f32, b: f32) {
    ///     ..
    /// }
    /// ```
    pub TOO_MANY_ARGUMENTS,
    complexity,
    "functions with too many arguments"
}

declare_clippy_lint! {
    /// **What it does:** Checks for functions with a large amount of lines.
    ///
    /// **Why is this bad?** Functions with a lot of lines are harder to understand
    /// due to having to look at a larger amount of code to understand what the
    /// function is doing. Consider splitting the body of the function into
    /// multiple functions.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ``` rust
    /// fn im_too_long() {
    /// println!("");
    /// // ... 100 more LoC
    /// println!("");
    /// }
    /// ```
    pub TOO_MANY_LINES,
    pedantic,
    "functions with too many lines"
}

declare_clippy_lint! {
    /// **What it does:** Checks for public functions that dereference raw pointer
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
    /// pub fn foo(x: *const u8) {
    ///     println!("{}", unsafe { *x });
    /// }
    /// ```
    pub NOT_UNSAFE_PTR_ARG_DEREF,
    correctness,
    "public functions dereferencing raw pointer arguments but not marked `unsafe`"
}

#[derive(Copy, Clone)]
pub struct Functions {
    threshold: u64,
    max_lines: u64,
}

impl Functions {
    pub fn new(threshold: u64, max_lines: u64) -> Self {
        Self { threshold, max_lines }
    }
}

impl_lint_pass!(Functions => [TOO_MANY_ARGUMENTS, TOO_MANY_LINES, NOT_UNSAFE_PTR_ARG_DEREF]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Functions {
    fn check_fn(
        &mut self,
        cx: &LateContext<'a, 'tcx>,
        kind: intravisit::FnKind<'tcx>,
        decl: &'tcx hir::FnDecl,
        body: &'tcx hir::Body,
        span: Span,
        hir_id: hir::HirId,
    ) {
        let is_impl = if let Some(hir::Node::Item(item)) = cx
            .tcx
            .hir()
            .find_by_hir_id(cx.tcx.hir().get_parent_node_by_hir_id(hir_id))
        {
            matches!(item.node, hir::ItemKind::Impl(_, _, _, _, Some(_), _, _))
        } else {
            false
        };

        let unsafety = match kind {
            hir::intravisit::FnKind::ItemFn(_, _, hir::FnHeader { unsafety, .. }, _, _) => unsafety,
            hir::intravisit::FnKind::Method(_, sig, _, _) => sig.header.unsafety,
            hir::intravisit::FnKind::Closure(_) => return,
        };

        // don't warn for implementations, it's not their fault
        if !is_impl {
            // don't lint extern functions decls, it's not their fault either
            match kind {
                hir::intravisit::FnKind::Method(
                    _,
                    &hir::MethodSig {
                        header: hir::FnHeader { abi: Abi::Rust, .. },
                        ..
                    },
                    _,
                    _,
                )
                | hir::intravisit::FnKind::ItemFn(_, _, hir::FnHeader { abi: Abi::Rust, .. }, _, _) => {
                    self.check_arg_number(cx, decl, span)
                },
                _ => {},
            }
        }

        self.check_raw_ptr(cx, unsafety, decl, body, hir_id);
        self.check_line_number(cx, span, body);
    }

    fn check_trait_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx hir::TraitItem) {
        if let hir::TraitItemKind::Method(ref sig, ref eid) = item.node {
            // don't lint extern functions decls, it's not their fault
            if sig.header.abi == Abi::Rust {
                self.check_arg_number(cx, &sig.decl, item.span);
            }

            if let hir::TraitMethod::Provided(eid) = *eid {
                let body = cx.tcx.hir().body(eid);
                self.check_raw_ptr(cx, sig.header.unsafety, &sig.decl, body, item.hir_id);
            }
        }
    }
}

impl<'a, 'tcx> Functions {
    fn check_arg_number(self, cx: &LateContext<'_, '_>, decl: &hir::FnDecl, span: Span) {
        // Remove the function body from the span. We can't use `SourceMap::def_span` because the
        // argument list might span multiple lines.
        let span = if let Some(snippet) = snippet_opt(cx, span) {
            let snippet = snippet.split('{').nth(0).unwrap_or("").trim_end();
            if snippet.is_empty() {
                span
            } else {
                span.with_hi(BytePos(span.lo().0 + u32::try_from(snippet.len()).unwrap()))
            }
        } else {
            span
        };

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

    fn check_line_number(self, cx: &LateContext<'_, '_>, span: Span, body: &'tcx hir::Body) {
        if in_external_macro(cx.sess(), span) {
            return;
        }

        let code_snippet = snippet(cx, body.value.span, "..");
        let mut line_count: u64 = 0;
        let mut in_comment = false;
        let mut code_in_line;

        // Skip the surrounding function decl.
        let start_brace_idx = match code_snippet.find('{') {
            Some(i) => i + 1,
            None => 0,
        };
        let end_brace_idx = match code_snippet.find('}') {
            Some(i) => i,
            None => code_snippet.len(),
        };
        let function_lines = code_snippet[start_brace_idx..end_brace_idx].lines();

        for mut line in function_lines {
            code_in_line = false;
            loop {
                line = line.trim_start();
                if line.is_empty() {
                    break;
                }
                if in_comment {
                    match line.find("*/") {
                        Some(i) => {
                            line = &line[i + 2..];
                            in_comment = false;
                            continue;
                        },
                        None => break,
                    }
                } else {
                    let multi_idx = match line.find("/*") {
                        Some(i) => i,
                        None => line.len(),
                    };
                    let single_idx = match line.find("//") {
                        Some(i) => i,
                        None => line.len(),
                    };
                    code_in_line |= multi_idx > 0 && single_idx > 0;
                    // Implies multi_idx is below line.len()
                    if multi_idx < single_idx {
                        line = &line[multi_idx + 2..];
                        in_comment = true;
                        continue;
                    }
                    break;
                }
            }
            if code_in_line {
                line_count += 1;
            }
        }

        if line_count > self.max_lines {
            span_lint(cx, TOO_MANY_LINES, span, "This function has a large number of lines.")
        }
    }

    fn check_raw_ptr(
        self,
        cx: &LateContext<'a, 'tcx>,
        unsafety: hir::Unsafety,
        decl: &'tcx hir::FnDecl,
        body: &'tcx hir::Body,
        hir_id: hir::HirId,
    ) {
        let expr = &body.value;
        if unsafety == hir::Unsafety::Normal && cx.access_levels.is_exported(hir_id) {
            let raw_ptrs = iter_input_pats(decl, body)
                .zip(decl.inputs.iter())
                .filter_map(|(arg, ty)| raw_ptr_arg(arg, ty))
                .collect::<FxHashSet<_>>();

            if !raw_ptrs.is_empty() {
                let tables = cx.tcx.body_tables(body.id());
                let mut v = DerefVisitor {
                    cx,
                    ptrs: raw_ptrs,
                    tables,
                };

                hir::intravisit::walk_expr(&mut v, expr);
            }
        }
    }
}

fn raw_ptr_arg(arg: &hir::Arg, ty: &hir::Ty) -> Option<hir::HirId> {
    if let (&hir::PatKind::Binding(_, id, _, _), &hir::TyKind::Ptr(_)) = (&arg.pat.node, &ty.node) {
        Some(id)
    } else {
        None
    }
}

struct DerefVisitor<'a, 'tcx: 'a> {
    cx: &'a LateContext<'a, 'tcx>,
    ptrs: FxHashSet<hir::HirId>,
    tables: &'a ty::TypeckTables<'tcx>,
}

impl<'a, 'tcx> hir::intravisit::Visitor<'tcx> for DerefVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx hir::Expr) {
        match expr.node {
            hir::ExprKind::Call(ref f, ref args) => {
                let ty = self.tables.expr_ty(f);

                if type_is_unsafe_function(self.cx, ty) {
                    for arg in args {
                        self.check_arg(arg);
                    }
                }
            },
            hir::ExprKind::MethodCall(_, _, ref args) => {
                let def_id = self.tables.type_dependent_def_id(expr.hir_id).unwrap();
                let base_type = self.cx.tcx.type_of(def_id);

                if type_is_unsafe_function(self.cx, base_type) {
                    for arg in args {
                        self.check_arg(arg);
                    }
                }
            },
            hir::ExprKind::Unary(hir::UnDeref, ref ptr) => self.check_arg(ptr),
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
        if let hir::ExprKind::Path(ref qpath) = ptr.node {
            if let Res::Local(id) = self.cx.tables.qpath_res(qpath, ptr.hir_id) {
                if self.ptrs.contains(&id) {
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
}
