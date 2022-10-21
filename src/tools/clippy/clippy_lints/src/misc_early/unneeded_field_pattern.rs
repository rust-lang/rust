use clippy_utils::diagnostics::{span_lint, span_lint_and_help};
use clippy_utils::source::snippet_opt;
use rustc_ast::ast::{Pat, PatKind};
use rustc_lint::EarlyContext;

use super::UNNEEDED_FIELD_PATTERN;

pub(super) fn check(cx: &EarlyContext<'_>, pat: &Pat) {
    if let PatKind::Struct(_, ref npat, ref pfields, _) = pat.kind {
        let mut wilds = 0;
        let type_name = npat
            .segments
            .last()
            .expect("A path must have at least one segment")
            .ident
            .name;

        for field in pfields {
            if let PatKind::Wild = field.pat.kind {
                wilds += 1;
            }
        }
        if !pfields.is_empty() && wilds == pfields.len() {
            span_lint_and_help(
                cx,
                UNNEEDED_FIELD_PATTERN,
                pat.span,
                "all the struct fields are matched to a wildcard pattern, consider using `..`",
                None,
                &format!("try with `{type_name} {{ .. }}` instead"),
            );
            return;
        }
        if wilds > 0 {
            for field in pfields {
                if let PatKind::Wild = field.pat.kind {
                    wilds -= 1;
                    if wilds > 0 {
                        span_lint(
                            cx,
                            UNNEEDED_FIELD_PATTERN,
                            field.span,
                            "you matched a field with a wildcard pattern, consider using `..` instead",
                        );
                    } else {
                        let mut normal = vec![];

                        for field in pfields {
                            match field.pat.kind {
                                PatKind::Wild => {},
                                _ => {
                                    if let Some(n) = snippet_opt(cx, field.span) {
                                        normal.push(n);
                                    }
                                },
                            }
                        }

                        span_lint_and_help(
                            cx,
                            UNNEEDED_FIELD_PATTERN,
                            field.span,
                            "you matched a field with a wildcard pattern, consider using `..` \
                             instead",
                            None,
                            &format!("try with `{type_name} {{ {}, .. }}`", normal[..].join(", ")),
                        );
                    }
                }
            }
        }
    }
}
