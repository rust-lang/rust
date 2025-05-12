use clippy_config::Conf;
use clippy_config::types::MacroMatcher;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::{SourceText, SpanRangeExt};
use rustc_ast::ast;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::Applicability;
use rustc_hir::def_id::DefId;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::impl_lint_pass;
use rustc_span::Span;
use rustc_span::hygiene::{ExpnKind, MacroKind};

declare_clippy_lint! {
    /// ### What it does
    /// Checks that common macros are used with consistent bracing.
    ///
    /// ### Why is this bad?
    /// This is mostly a consistency lint although using () or []
    /// doesn't give you a semicolon in item position, which can be unexpected.
    ///
    /// ### Example
    /// ```no_run
    /// vec!{1, 2, 3};
    /// ```
    /// Use instead:
    /// ```no_run
    /// vec![1, 2, 3];
    /// ```
    #[clippy::version = "1.55.0"]
    pub NONSTANDARD_MACRO_BRACES,
    nursery,
    "check consistent use of braces in macro"
}

/// The (callsite span, (open brace, close brace), source snippet)
type MacroInfo = (Span, (char, char), SourceText);

pub struct MacroBraces {
    macro_braces: FxHashMap<String, (char, char)>,
    done: FxHashSet<Span>,
}

impl MacroBraces {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            macro_braces: macro_braces(&conf.standard_macro_braces),
            done: FxHashSet::default(),
        }
    }
}

impl_lint_pass!(MacroBraces => [NONSTANDARD_MACRO_BRACES]);

impl EarlyLintPass for MacroBraces {
    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &ast::Item) {
        if let Some((span, braces, snip)) = is_offending_macro(cx, item.span, self) {
            emit_help(cx, &snip, braces, span);
            self.done.insert(span);
        }
    }

    fn check_stmt(&mut self, cx: &EarlyContext<'_>, stmt: &ast::Stmt) {
        if let Some((span, braces, snip)) = is_offending_macro(cx, stmt.span, self) {
            emit_help(cx, &snip, braces, span);
            self.done.insert(span);
        }
    }

    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &ast::Expr) {
        if let Some((span, braces, snip)) = is_offending_macro(cx, expr.span, self) {
            emit_help(cx, &snip, braces, span);
            self.done.insert(span);
        }
    }

    fn check_ty(&mut self, cx: &EarlyContext<'_>, ty: &ast::Ty) {
        if let Some((span, braces, snip)) = is_offending_macro(cx, ty.span, self) {
            emit_help(cx, &snip, braces, span);
            self.done.insert(span);
        }
    }
}

fn is_offending_macro(cx: &EarlyContext<'_>, span: Span, mac_braces: &MacroBraces) -> Option<MacroInfo> {
    let unnested_or_local = || {
        !span.ctxt().outer_expn_data().call_site.from_expansion()
            || span
                .macro_backtrace()
                .last()
                .is_some_and(|e| e.macro_def_id.is_some_and(DefId::is_local))
    };
    let span_call_site = span.ctxt().outer_expn_data().call_site;
    if let ExpnKind::Macro(MacroKind::Bang, mac_name) = span.ctxt().outer_expn_data().kind
        && let name = mac_name.as_str()
        && let Some(&braces) = mac_braces.macro_braces.get(name)
        && let Some(snip) = span_call_site.get_source_text(cx)
        // we must check only invocation sites
        // https://github.com/rust-lang/rust-clippy/issues/7422
        && snip.starts_with(&format!("{name}!"))
        && unnested_or_local()
        // make formatting consistent
        && let c = snip.replace(' ', "")
        && !c.starts_with(&format!("{name}!{}", braces.0))
        && !mac_braces.done.contains(&span_call_site)
    {
        Some((span_call_site, braces, snip))
    } else {
        None
    }
}

fn emit_help(cx: &EarlyContext<'_>, snip: &str, (open, close): (char, char), span: Span) {
    if let Some((macro_name, macro_args_str)) = snip.split_once('!') {
        let mut macro_args = macro_args_str.trim().to_string();
        // now remove the wrong braces
        macro_args.remove(0);
        macro_args.pop();
        span_lint_and_sugg(
            cx,
            NONSTANDARD_MACRO_BRACES,
            span,
            format!("use of irregular braces for `{macro_name}!` macro"),
            "consider writing",
            format!("{macro_name}!{open}{macro_args}{close}"),
            Applicability::MachineApplicable,
        );
    }
}

fn macro_braces(conf: &[MacroMatcher]) -> FxHashMap<String, (char, char)> {
    let mut braces = FxHashMap::from_iter(
        [
            ("print", ('(', ')')),
            ("println", ('(', ')')),
            ("eprint", ('(', ')')),
            ("eprintln", ('(', ')')),
            ("write", ('(', ')')),
            ("writeln", ('(', ')')),
            ("format", ('(', ')')),
            ("format_args", ('(', ')')),
            ("vec", ('[', ']')),
            ("matches", ('(', ')')),
        ]
        .map(|(k, v)| (k.to_string(), v)),
    );
    // We want users items to override any existing items
    for it in conf {
        braces.insert(it.name.clone(), it.braces);
    }
    braces
}
