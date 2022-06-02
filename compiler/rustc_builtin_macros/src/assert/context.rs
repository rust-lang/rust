use rustc_ast::{ptr::P, Expr, Path};
use rustc_expand::base::ExtCtxt;
use rustc_span::Span;

pub(super) struct Context<'cx, 'a> {
    cx: &'cx ExtCtxt<'a>,
    span: Span,
}

impl<'cx, 'a> Context<'cx, 'a> {
    pub(super) fn new(cx: &'cx ExtCtxt<'a>, span: Span) -> Self {
        Self { cx, span }
    }

    /// Builds the whole `assert!` expression.
    ///
    /// {
    ///    use ::core::asserting::{ ... };
    ///
    ///    let mut __capture0 = Capture::new();
    ///    ...
    ///    ...
    ///    ...
    ///
    ///    if !{
    ///       ...
    ///       ...
    ///       ...
    ///    } {
    ///        panic!(
    ///            "Assertion failed: ... \n With expansion: ...",
    ///            __capture0,
    ///            ...
    ///            ...
    ///            ...
    ///        );
    ///    }
    /// }
    pub(super) fn build(self, _cond_expr: P<Expr>, _panic_path: Path) -> P<Expr> {
        let Self { cx, span, .. } = self;
        let stmts = Vec::new();
        cx.expr_block(cx.block(span, stmts))
    }
}
