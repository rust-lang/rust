use rustc_hir as hir;
use rustc_hir::intravisit::FnKind;
use rustc_lint::{LateContext, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_span::Span;

use clippy_utils::diagnostics::span_lint;
use clippy_utils::source::snippet_opt;

use super::TOO_MANY_LINES;

pub(super) fn check_fn(
    cx: &LateContext<'_>,
    kind: FnKind<'tcx>,
    span: Span,
    body: &'tcx hir::Body<'_>,
    too_many_lines_threshold: u64,
) {
    // Closures must be contained in a parent body, which will be checked for `too_many_lines`.
    // Don't check closures for `too_many_lines` to avoid duplicated lints.
    if matches!(kind, FnKind::Closure) || in_external_macro(cx.sess(), span) {
        return;
    }

    let code_snippet = match snippet_opt(cx, body.value.span) {
        Some(s) => s,
        _ => return,
    };
    let mut line_count: u64 = 0;
    let mut in_comment = false;
    let mut code_in_line;

    let function_lines = if matches!(body.value.kind, hir::ExprKind::Block(..))
        && code_snippet.as_bytes().first().copied() == Some(b'{')
        && code_snippet.as_bytes().last().copied() == Some(b'}')
    {
        // Removing the braces from the enclosing block
        &code_snippet[1..code_snippet.len() - 1]
    } else {
        &code_snippet
    }
    .trim() // Remove leading and trailing blank lines
    .lines();

    for mut line in function_lines {
        code_in_line = false;
        loop {
            line = line.trim_start();
            if line.is_empty() {
                break;
            }
            if in_comment {
                if let Some(i) = line.find("*/") {
                    line = &line[i + 2..];
                    in_comment = false;
                    continue;
                }
            } else {
                let multi_idx = line.find("/*").unwrap_or(line.len());
                let single_idx = line.find("//").unwrap_or(line.len());
                code_in_line |= multi_idx > 0 && single_idx > 0;
                // Implies multi_idx is below line.len()
                if multi_idx < single_idx {
                    line = &line[multi_idx + 2..];
                    in_comment = true;
                    continue;
                }
            }
            break;
        }
        if code_in_line {
            line_count += 1;
        }
    }

    if line_count > too_many_lines_threshold {
        span_lint(
            cx,
            TOO_MANY_LINES,
            span,
            &format!(
                "this function has too many lines ({}/{})",
                line_count, too_many_lines_threshold
            ),
        );
    }
}
