use rustc_ast::{LitKind, StrStyle};
use rustc_hir::{Expr, ExprKind};
use rustc_lexer::is_ident;
use rustc_lint::{LateContext, LateLintPass};
use rustc_parse_format::{ParseMode, Parser, Piece};
use rustc_session::declare_lint_pass;
use rustc_span::{BytePos, Span};

use clippy_utils::diagnostics::span_lint;
use clippy_utils::is_from_proc_macro;
use clippy_utils::mir::enclosing_mir;

declare_clippy_lint! {
    /// ### What it does
    /// Checks if string literals have formatting arguments outside of macros
    /// using them (like `format!`).
    ///
    /// ### Why is this bad?
    /// It will likely not generate the expected content.
    ///
    /// ### Example
    /// ```no_run
    /// let x: Option<usize> = None;
    /// let y = "hello";
    /// x.expect("{y:?}");
    /// ```
    /// Use instead:
    /// ```no_run
    /// let x: Option<usize> = None;
    /// let y = "hello";
    /// x.expect(&format!("{y:?}"));
    /// ```
    #[clippy::version = "1.85.0"]
    pub LITERAL_STRING_WITH_FORMATTING_ARGS,
    nursery,
    "Checks if string literals have formatting arguments"
}

declare_lint_pass!(LiteralStringWithFormattingArg => [LITERAL_STRING_WITH_FORMATTING_ARGS]);

fn emit_lint(cx: &LateContext<'_>, expr: &Expr<'_>, spans: &[(Span, Option<String>)]) {
    if !spans.is_empty()
        && let Some(mir) = enclosing_mir(cx.tcx, expr.hir_id)
    {
        let spans = spans
            .iter()
            .filter_map(|(span, name)| {
                if let Some(name) = name
                    // We need to check that the name is a local.
                    && !mir
                        .var_debug_info
                        .iter()
                        .any(|local| !local.source_info.span.from_expansion() && local.name.as_str() == name)
                {
                    return None;
                }
                Some(*span)
            })
            .collect::<Vec<_>>();
        match spans.len() {
            0 => {},
            1 => {
                span_lint(
                    cx,
                    LITERAL_STRING_WITH_FORMATTING_ARGS,
                    spans,
                    "this looks like a formatting argument but it is not part of a formatting macro",
                );
            },
            _ => {
                span_lint(
                    cx,
                    LITERAL_STRING_WITH_FORMATTING_ARGS,
                    spans,
                    "these look like formatting arguments but are not part of a formatting macro",
                );
            },
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for LiteralStringWithFormattingArg {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &Expr<'tcx>) {
        if expr.span.from_expansion() || expr.span.is_dummy() {
            return;
        }
        if let ExprKind::Lit(lit) = expr.kind {
            let (add, symbol) = match lit.node {
                LitKind::Str(symbol, style) => {
                    let add = match style {
                        StrStyle::Cooked => 1,
                        StrStyle::Raw(nb) => nb as usize + 2,
                    };
                    (add, symbol)
                },
                _ => return,
            };
            if is_from_proc_macro(cx, expr) {
                return;
            }
            let fmt_str = symbol.as_str();
            let lo = expr.span.lo();
            let mut current = fmt_str;
            let mut diff_len = 0;

            let mut parser = Parser::new(current, None, None, false, ParseMode::Format);
            let mut spans = Vec::new();
            while let Some(piece) = parser.next() {
                if let Some(error) = parser.errors.last() {
                    // We simply ignore the errors and move after them.
                    if error.span.end >= current.len() {
                        break;
                    }
                    // We find the closest char to where the error location ends.
                    let pos = current.floor_char_boundary(error.span.end);
                    // We get the next character.
                    current = if let Some((next_char_pos, _)) = current[pos..].char_indices().nth(1) {
                        // We make the parser start from this new location.
                        &current[pos + next_char_pos..]
                    } else {
                        break;
                    };
                    diff_len = fmt_str.len() - current.len();
                    parser = Parser::new(current, None, None, false, ParseMode::Format);
                } else if let Piece::NextArgument(arg) = piece {
                    let mut pos = arg.position_span;
                    pos.start += diff_len;
                    pos.end += diff_len;

                    let mut start = pos.start;
                    while start < fmt_str.len() && !fmt_str.is_char_boundary(start) {
                        start += 1;
                    }
                    let start = fmt_str[..start].rfind('{').unwrap_or(start);
                    // If this is a unicode character escape, we don't want to lint.
                    if start > 1 && fmt_str[..start].ends_with("\\u") {
                        continue;
                    }

                    if fmt_str[start + 1..].trim_start().starts_with('}') {
                        // We ignore `{}`.
                        continue;
                    }

                    let end = fmt_str[start + 1..]
                        .find('}')
                        .map_or(pos.end, |found| start + 1 + found)
                        + 1;
                    let ident_start = start + 1;
                    let colon_pos = fmt_str[ident_start..end].find(':');
                    let ident_end = colon_pos.unwrap_or(end - 1);
                    let mut name = None;
                    if ident_start < ident_end
                        && let arg = &fmt_str[ident_start..ident_end]
                        && !arg.is_empty()
                        && is_ident(arg)
                    {
                        name = Some(arg.to_string());
                    } else if colon_pos.is_none() {
                        // Not a `{:?}`.
                        continue;
                    }
                    spans.push((
                        expr.span
                            .with_hi(lo + BytePos((start + add).try_into().unwrap()))
                            .with_lo(lo + BytePos((end + add).try_into().unwrap())),
                        name,
                    ));
                }
            }
            emit_lint(cx, expr, &spans);
        }
    }
}
