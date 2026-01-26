use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::res::MaybeDef;
use clippy_utils::source::snippet_with_context;
use clippy_utils::visitors::is_expr_unsafe;
use clippy_utils::{match_libc_symbol, sym};
use rustc_errors::Applicability;
use rustc_hir::{Block, BlockCheckMode, Expr, ExprKind, LangItem, Node, UnsafeSource};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::impl_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `libc::strlen` on a `CString` or `CStr` value,
    /// and suggest calling `count_bytes()` instead.
    ///
    /// ### Why is this bad?
    /// libc::strlen is an unsafe function, which we don't need to call
    /// if all we want to know is the length of the c-string.
    ///
    /// ### Example
    /// ```rust, ignore
    /// use std::ffi::CString;
    /// let cstring = CString::new("foo").expect("CString::new failed");
    /// let len = unsafe { libc::strlen(cstring.as_ptr()) };
    /// ```
    /// Use instead:
    /// ```rust, no_run
    /// use std::ffi::CString;
    /// let cstring = CString::new("foo").expect("CString::new failed");
    /// let len = cstring.count_bytes();
    /// ```
    #[clippy::version = "1.55.0"]
    pub STRLEN_ON_C_STRINGS,
    complexity,
    "using `libc::strlen` on a `CString` or `CStr` value, while `count_bytes()` can be used instead"
}

pub struct StrlenOnCStrings {
    msrv: Msrv,
}

impl StrlenOnCStrings {
    pub fn new(conf: &Conf) -> Self {
        Self { msrv: conf.msrv }
    }
}

impl_lint_pass!(StrlenOnCStrings => [STRLEN_ON_C_STRINGS]);

impl<'tcx> LateLintPass<'tcx> for StrlenOnCStrings {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if !expr.span.from_expansion()
            && let ExprKind::Call(func, [recv]) = expr.kind
            && let ExprKind::Path(path) = &func.kind
            && let Some(did) = cx.qpath_res(path, func.hir_id).opt_def_id()
            && match_libc_symbol(cx, did, sym::strlen)
            && let ExprKind::MethodCall(path, self_arg, [], _) = recv.kind
            && !recv.span.from_expansion()
            && path.ident.name == sym::as_ptr
            && let typeck = cx.typeck_results()
            && typeck
                .expr_ty_adjusted(self_arg)
                .peel_refs()
                .is_lang_item(cx, LangItem::CStr)
        {
            let ty = typeck.expr_ty(self_arg).peel_refs();
            let ty_kind = if ty.is_diag_item(cx, sym::cstring_type) {
                "`CString` value"
            } else if ty.is_lang_item(cx, LangItem::CStr) {
                "`CStr` value"
            } else {
                "type that dereferences to `CStr`"
            };

            let ctxt = expr.span.ctxt();
            let span = match cx.tcx.parent_hir_node(expr.hir_id) {
                Node::Block(&Block {
                    rules: BlockCheckMode::UnsafeBlock(UnsafeSource::UserProvided),
                    span,
                    ..
                }) if span.ctxt() == ctxt && !is_expr_unsafe(cx, self_arg) => span,
                _ => expr.span,
            };

            span_lint_and_then(
                cx,
                STRLEN_ON_C_STRINGS,
                span,
                format!("using `libc::strlen` on a {ty_kind}"),
                |diag| {
                    let mut app = Applicability::MachineApplicable;
                    let val_name = snippet_with_context(cx, self_arg.span, ctxt, "_", &mut app).0;

                    let suggestion = if self.msrv.meets(cx, msrvs::CSTR_COUNT_BYTES) {
                        format!("{val_name}.count_bytes()")
                    } else {
                        format!("{val_name}.to_bytes().len()")
                    };

                    diag.span_suggestion(span, "use", suggestion, app);
                },
            );
        }
    }
}
