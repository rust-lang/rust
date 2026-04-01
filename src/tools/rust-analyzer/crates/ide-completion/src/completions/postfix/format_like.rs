// Feature: Format String Completion
//
// `"Result {result} is {2 + 2}"` is expanded to the `"Result {} is {}", result, 2 + 2`.
//
// The following postfix snippets are available:
//
// * `format` -> `format!(...)`
// * `panic` -> `panic!(...)`
// * `println` -> `println!(...)`
// * `log`:
// ** `logd` -> `log::debug!(...)`
// ** `logt` -> `log::trace!(...)`
// ** `logi` -> `log::info!(...)`
// ** `logw` -> `log::warn!(...)`
// ** `loge` -> `log::error!(...)`
//
// ![Format String Completion](https://user-images.githubusercontent.com/48062697/113020656-b560f500-917a-11eb-87de-02991f61beb8.gif)

use ide_db::{
    SnippetCap,
    syntax_helpers::format_string_exprs::{Arg, parse_format_exprs, with_placeholders},
};
use syntax::{AstToken, ast};

use crate::{
    Completions,
    completions::postfix::{build_postfix_snippet_builder, escape_snippet_bits},
    context::CompletionContext,
};

/// Mapping ("postfix completion item" => "macro to use")
static KINDS: &[(&str, &str)] = &[
    ("format", "format!"),
    ("panic", "panic!"),
    ("println", "println!"),
    ("eprintln", "eprintln!"),
    ("logd", "log::debug!"),
    ("logt", "log::trace!"),
    ("logi", "log::info!"),
    ("logw", "log::warn!"),
    ("loge", "log::error!"),
];

pub(crate) fn add_format_like_completions(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    dot_receiver: &ast::Expr,
    cap: SnippetCap,
    receiver_text: &ast::String,
) {
    let postfix_snippet = match build_postfix_snippet_builder(ctx, cap, dot_receiver) {
        Some(it) => it,
        None => return,
    };

    if let Ok((mut out, mut exprs)) = parse_format_exprs(receiver_text.text()) {
        // Escape any snippet bits in the out text and any of the exprs.
        escape_snippet_bits(&mut out);
        for arg in &mut exprs {
            if let Arg::Ident(text) | Arg::Expr(text) = arg {
                escape_snippet_bits(text)
            }
        }

        let exprs = with_placeholders(exprs);
        for (label, macro_name) in KINDS {
            let snippet = if exprs.is_empty() {
                format!(r#"{macro_name}({out})"#)
            } else {
                format!(r#"{}({}, {})"#, macro_name, out, exprs.join(", "))
            };

            postfix_snippet(label, macro_name, &snippet).add_to(acc, ctx.db);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_into_suggestion() {
        let test_vector = &[
            ("println!", "{}", r#"println!("{}", $1)"#),
            ("eprintln!", "{}", r#"eprintln!("{}", $1)"#),
            (
                "log::info!",
                "{} {ident} {} {2 + 2}",
                r#"log::info!("{} {ident} {} {}", $1, $2, 2 + 2)"#,
            ),
        ];

        for (kind, input, output) in test_vector {
            let (parsed_string, exprs) = parse_format_exprs(input).unwrap();
            let exprs = with_placeholders(exprs);
            let snippet = format!(r#"{kind}("{parsed_string}", {})"#, exprs.join(", "));
            assert_eq!(&snippet, output);
        }
    }

    #[test]
    fn test_into_suggestion_no_epxrs() {
        let test_vector = &[
            ("println!", "{ident}", r#"println!("{ident}")"#),
            ("format!", "{ident:?}", r#"format!("{ident:?}")"#),
        ];

        for (kind, input, output) in test_vector {
            let (parsed_string, _exprs) = parse_format_exprs(input).unwrap();
            let snippet = format!(r#"{kind}("{parsed_string}")"#);
            assert_eq!(&snippet, output);
        }
    }
}
