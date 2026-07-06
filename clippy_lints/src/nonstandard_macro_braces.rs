use clippy_config::Conf;
use clippy_config::types::MacroMatcher;
use clippy_utils::diagnostics::span_lint_and_sugg;
use rustc_ast::ast;
use rustc_ast::token::{Delimiter, Token, TokenKind};
use rustc_ast::tokenstream::{TokenStream, TokenTree};
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::impl_lint_pass;
use rustc_span::Span;

use crate::rustc_lint::LintContext as _;
use clippy_utils::source::snippet_opt;

declare_clippy_lint! {
    /// ### What it does
    /// Checks that common macros are used with consistent bracing.
    ///
    /// ### Why is this bad?
    /// Having non-conventional braces on well-stablished macros can be confusing
    /// when debugging, and they bring incosistencies with the rest of the ecosystem.
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
    style,
    "check consistent use of braces in macro"
}

impl_lint_pass!(MacroBraces => [NONSTANDARD_MACRO_BRACES]);

pub struct MacroBraces {
    macro_braces: (FxHashMap<String, (char, char)>, usize),
    /// Spans for statement macro calls, they have special behaviour with semicolons
    mac_stmt_spans: Vec<Span>,
}

impl MacroBraces {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            macro_braces: macro_braces(&conf.standard_macro_braces),
            mac_stmt_spans: Vec::new(),
        }
    }
}

impl EarlyLintPass for MacroBraces {
    fn check_mac(&mut self, cx: &EarlyContext<'_>, mac: &ast::MacCall) {
        if let Some(last_segment) = mac.path.segments.last()
            && let name = last_segment.ident.as_str()
            && let Some(&braces) = self.macro_braces.0.get(name)
            && let Some(snip) = snippet_opt(cx.sess(), mac.span().with_lo(last_segment.span().lo()))
            && let Some(macro_args_str) = &snip.strip_prefix(name).and_then(|snip| snip.strip_prefix('!'))
            && let Some(old_open_brace @ ('{' | '(' | '[')) = macro_args_str.trim_start().chars().next()
            && old_open_brace != braces.0
        {
            // Semicolons added for statements that previously ended in braces, see issue #9913
            let add_semi = self.mac_stmt_spans.iter().any(|s| *s == mac.span());
            emit_help(
                cx,
                &snippet_opt(cx.sess(), mac.span()).unwrap(),
                braces,
                mac.span(),
                add_semi,
            );
        }
    }

    // See issue #9913
    fn check_stmt(&mut self, _: &EarlyContext<'_>, stmt: &ast::Stmt) {
        if let ast::StmtKind::MacCall(mac_callstmt) = &stmt.kind
            && let ast::MacCallStmt {
                style: ast::MacStmtStyle::Braces,
                ..
            } = **mac_callstmt
        {
            self.mac_stmt_spans.push(mac_callstmt.mac.span());
        }
    }

    fn check_mac_def(&mut self, cx: &EarlyContext<'_>, mac: &ast::MacroDef) {
        fn check_ts(cx: &EarlyContext<'_>, ts: &TokenStream, macro_braces: &FxHashMap<String, (char, char)>) {
            for (i, current_token) in ts.iter().enumerate() {
                if let TokenTree::Delimited(_, _, _, token_stream) = current_token {
                    // Peel extra braces and parenthesis in macros!
                    check_ts(cx, token_stream, macro_braces);
                } else
                //        |-TokenKind::Bang
                //        v
                // println! { "Hi" }
                // ^^^^^^^
                //    |     ^^^^^^^^ Brackets always come 1 token after TokenKind::Bang
                // ident_token
                if let TokenTree::Token(
                    Token {
                        kind: TokenKind::Ident(ident_token, _),
                        span: ident_span,
                    },
                    _,
                ) = current_token
                    && let Some(bang_token) = ts.get(i + 1)
                    && let Some(macro_args_token) = ts.get(i + 2)
                    && let TokenTree::Token(
                        Token {
                            kind: TokenKind::Bang, ..
                        },
                        _,
                    ) = *bang_token
                    && let TokenTree::Delimited(delim_span, _, delim, _) = macro_args_token
                    // Span from ident_token to brackets (so, the full macro call)
                    && let snip_span = ident_span.with_hi(delim_span.close.hi())
                    && let Some(snip) = snippet_opt(cx, snip_span)
                    && let Some(&braces) = macro_braces.get(ident_token.as_str())
                    && let Some(old_open_brace) = match delim {
                        Delimiter::Brace => Some('{'),
                        Delimiter::Parenthesis => Some('('),
                        Delimiter::Bracket => Some('['),
                        Delimiter::Invisible(_) => None,
                    }
                    && old_open_brace != braces.0
                {
                    emit_help(cx, &snip, braces, snip_span, false);
                }
            }
        }

        if mac.macro_rules {
            check_ts(cx, &mac.body.tokens, &self.macro_braces.0);
        }
    }
}

fn emit_help(cx: &EarlyContext<'_>, snip: &str, (open, close): (char, char), span: Span, add_semi: bool) {
    let semi = if add_semi { ";" } else { "" };
    if let Some((macro_name, macro_args_str)) = snip.split_once('!') {
        let mut macro_args = macro_args_str.trim().to_string();
        // now remove the wrong braces
        macro_args.pop();
        macro_args.remove(0);
        span_lint_and_sugg(
            cx,
            NONSTANDARD_MACRO_BRACES,
            span,
            format!("use of irregular braces for `{macro_name}!` macro"),
            "consider writing",
            format!("{macro_name}!{open}{macro_args}{close}{semi}"),
            Applicability::MachineApplicable,
        );
    }
}

fn macro_braces(conf: &[MacroMatcher]) -> (FxHashMap<String, (char, char)>, usize) {
    // TODO: Use `Symbol`s here, instead of strings.
    let mut braces = FxHashMap::from_iter(
        [
            ("assert_matches", ('(', ')')),
            ("cfg_select", ('{', '}')),
            ("debug_assert_matches", ('(', ')')),
            ("format", ('(', ')')),
            ("format_args", ('(', ')')),
            ("eprint", ('(', ')')),
            ("eprintln", ('(', ')')),
            ("print", ('(', ')')),
            ("println", ('(', ')')),
            ("write", ('(', ')')),
            ("writeln", ('(', ')')),
            ("vec", ('[', ']')),
            ("matches", ('(', ')')),
        ]
        .map(|(k, v)| (k.to_string(), v)),
    );
    // We want users items to override any existing items
    for it in conf {
        braces.insert(it.name.clone(), it.braces);
    }

    #[expect(
        rustc::potential_query_instability,
        reason = "iteration order does not matter for `.max()`"
    )]
    #[expect(clippy::redundant_closure_for_method_calls, reason = "Clarity")]
    let max_len = braces
        .keys()
        .map(|macro_name| macro_name.len())
        .max()
        .expect("`braces` is non-empty");

    (braces, max_len)
}
