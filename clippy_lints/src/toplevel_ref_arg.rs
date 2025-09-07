use clippy_utils::diagnostics::{span_lint_hir, span_lint_hir_and_then};
use clippy_utils::source::{snippet, snippet_with_context};
use clippy_utils::sugg::Sugg;
use clippy_utils::{is_lint_allowed, iter_input_pats};
use rustc_errors::Applicability;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{BindingMode, Body, ByRef, FnDecl, Mutability, PatKind, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::Span;
use rustc_span::def_id::LocalDefId;

use crate::ref_patterns::REF_PATTERNS;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for function arguments and let bindings denoted as
    /// `ref`.
    ///
    /// ### Why is this bad?
    /// The `ref` declaration makes the function take an owned
    /// value, but turns the argument into a reference (which means that the value
    /// is destroyed when exiting the function). This adds not much value: either
    /// take a reference type, or take an owned value and create references in the
    /// body.
    ///
    /// For let bindings, `let x = &foo;` is preferred over `let ref x = foo`. The
    /// type of `x` is more obvious with the former.
    ///
    /// ### Known problems
    /// If the argument is dereferenced within the function,
    /// removing the `ref` will lead to errors. This can be fixed by removing the
    /// dereferences, e.g., changing `*x` to `x` within the function.
    ///
    /// ### Example
    /// ```no_run
    /// fn foo(ref _x: u8) {}
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// fn foo(_x: &u8) {}
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub TOPLEVEL_REF_ARG,
    style,
    "an entire binding declared as `ref`, in a function argument or a `let` statement"
}

declare_lint_pass!(ToplevelRefArg => [TOPLEVEL_REF_ARG]);

impl<'tcx> LateLintPass<'tcx> for ToplevelRefArg {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        k: FnKind<'tcx>,
        decl: &'tcx FnDecl<'_>,
        body: &'tcx Body<'_>,
        _: Span,
        _: LocalDefId,
    ) {
        if !matches!(k, FnKind::Closure) {
            for arg in iter_input_pats(decl, body) {
                if let PatKind::Binding(BindingMode(ByRef::Yes(_), _), ..) = arg.pat.kind
                    && is_lint_allowed(cx, REF_PATTERNS, arg.pat.hir_id)
                    && !arg.span.in_external_macro(cx.tcx.sess.source_map())
                {
                    span_lint_hir(
                        cx,
                        TOPLEVEL_REF_ARG,
                        arg.hir_id,
                        arg.pat.span,
                        "`ref` directly on a function parameter does not prevent taking ownership of the passed argument. \
                            Consider using a reference type instead",
                    );
                }
            }
        }
    }

    fn check_stmt(&mut self, cx: &LateContext<'tcx>, stmt: &'tcx Stmt<'_>) {
        if let StmtKind::Let(local) = stmt.kind
            && let PatKind::Binding(BindingMode(ByRef::Yes(mutabl), _), .., name, None) = local.pat.kind
            && let Some(init) = local.init
            // Do not emit if clippy::ref_patterns is not allowed to avoid having two lints for the same issue.
            && is_lint_allowed(cx, REF_PATTERNS, local.pat.hir_id)
            && !stmt.span.in_external_macro(cx.tcx.sess.source_map())
        {
            let ctxt = local.span.ctxt();
            let mut app = Applicability::MachineApplicable;
            let sugg_init = Sugg::hir_with_context(cx, init, ctxt, "..", &mut app);
            let (mutopt, initref) = match mutabl {
                Mutability::Mut => ("mut ", sugg_init.mut_addr()),
                Mutability::Not => ("", sugg_init.addr()),
            };
            let tyopt = if let Some(ty) = local.ty {
                let ty_snip = snippet_with_context(cx, ty.span, ctxt, "_", &mut app).0;
                format!(": &{mutopt}{ty_snip}")
            } else {
                String::new()
            };
            span_lint_hir_and_then(
                cx,
                TOPLEVEL_REF_ARG,
                init.hir_id,
                local.pat.span,
                "`ref` on an entire `let` pattern is discouraged, take a reference with `&` instead",
                |diag| {
                    diag.span_suggestion(
                        stmt.span,
                        "try",
                        format!("let {name}{tyopt} = {initref};", name = snippet(cx, name.span, ".."),),
                        app,
                    );
                },
            );
        }
    }
}
