//! Postfix completion for `format`-like strings.
//!
//! `"Result {result} is {2 + 2}"` is expanded to the `"Result {} is {}", result, 2 + 2`.
//!
//! The following postfix snippets are available:
//!
//! - `format` -> `format!(...)`
//! - `panic` -> `panic!(...)`
//! - `println` -> `println!(...)`
//! - `log`:
//!   + `logd` -> `log::debug!(...)`
//!   + `logt` -> `log::trace!(...)`
//!   + `logi` -> `log::info!(...)`
//!   + `logw` -> `log::warn!(...)`
//!   + `loge` -> `log::error!(...)`

use super::postfix_snippet;
use crate::completion::{
    completion_config::SnippetCap, completion_context::CompletionContext,
    completion_item::Completions,
};
use syntax::ast;

pub(super) fn add_format_like_completions(
    acc: &mut Completions,
    ctx: &CompletionContext,
    dot_receiver: &ast::Expr,
    cap: SnippetCap,
    receiver_text: &str,
) {
    if !is_string_literal(receiver_text) {
        // It's not a string literal, do not parse input.
        return;
    }

    let input = &receiver_text[1..receiver_text.len() - 1];

    let mut parser = FormatStrParser::new(input);

    if parser.parse().is_ok() {
        for kind in PostfixKind::all_suggestions() {
            let snippet = parser.into_suggestion(*kind);
            let (label, detail) = kind.into_description();

            postfix_snippet(ctx, cap, &dot_receiver, label, detail, &snippet).add_to(acc);
        }
    }
}

/// Checks whether provided item is a string literal.
fn is_string_literal(item: &str) -> bool {
    if item.len() < 2 {
        return false;
    }
    item.starts_with("\"") && item.ends_with("\"")
}

/// Parser for a format-like string. It is more allowing in terms of string contents,
/// as we expect variable placeholders to be filled with expressions.
#[derive(Debug)]
pub struct FormatStrParser {
    input: String,
    output: String,
    extracted_expressions: Vec<String>,
    state: State,
    parsed: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum PostfixKind {
    Format,
    Panic,
    Println,
    LogDebug,
    LogTrace,
    LogInfo,
    LogWarn,
    LogError,
}

impl PostfixKind {
    pub fn all_suggestions() -> &'static [PostfixKind] {
        &[
            Self::Format,
            Self::Panic,
            Self::Println,
            Self::LogDebug,
            Self::LogTrace,
            Self::LogInfo,
            Self::LogWarn,
            Self::LogError,
        ]
    }

    pub fn into_description(self) -> (&'static str, &'static str) {
        match self {
            Self::Format => ("fmt", "format!"),
            Self::Panic => ("panic", "panic!"),
            Self::Println => ("println", "println!"),
            Self::LogDebug => ("logd", "log::debug!"),
            Self::LogTrace => ("logt", "log::trace!"),
            Self::LogInfo => ("logi", "log::info!"),
            Self::LogWarn => ("logw", "log::warn!"),
            Self::LogError => ("loge", "log::error!"),
        }
    }

    pub fn into_macro_name(self) -> &'static str {
        match self {
            Self::Format => "format!",
            Self::Panic => "panic!",
            Self::Println => "println!",
            Self::LogDebug => "log::debug!",
            Self::LogTrace => "log::trace!",
            Self::LogInfo => "log::info!",
            Self::LogWarn => "log::warn!",
            Self::LogError => "log::error!",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum State {
    NotExpr,
    MaybeExpr,
    Expr,
    MaybeIncorrect,
    FormatOpts,
}

impl FormatStrParser {
    pub fn new(input: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            output: String::new(),
            extracted_expressions: Vec::new(),
            state: State::NotExpr,
            parsed: false,
        }
    }

    pub fn parse(&mut self) -> Result<(), ()> {
        let mut current_expr = String::new();

        let mut placeholder_id = 1;

        // Count of open braces inside of an expression.
        // We assume that user knows what they're doing, thus we treat it like a correct pattern, e.g.
        // "{MyStruct { val_a: 0, val_b: 1 }}".
        let mut inexpr_open_count = 0;

        for chr in self.input.chars() {
            match (self.state, chr) {
                (State::NotExpr, '{') => {
                    self.output.push(chr);
                    self.state = State::MaybeExpr;
                }
                (State::NotExpr, '}') => {
                    self.output.push(chr);
                    self.state = State::MaybeIncorrect;
                }
                (State::NotExpr, _) => {
                    self.output.push(chr);
                }
                (State::MaybeIncorrect, '}') => {
                    // It's okay, we met "}}".
                    self.output.push(chr);
                    self.state = State::NotExpr;
                }
                (State::MaybeIncorrect, _) => {
                    // Error in the string.
                    return Err(());
                }
                (State::MaybeExpr, '{') => {
                    self.output.push(chr);
                    self.state = State::NotExpr;
                }
                (State::MaybeExpr, '}') => {
                    // This is an empty sequence '{}'. Replace it with placeholder.
                    self.output.push(chr);
                    self.extracted_expressions.push(format!("${}", placeholder_id));
                    placeholder_id += 1;
                    self.state = State::NotExpr;
                }
                (State::MaybeExpr, _) => {
                    current_expr.push(chr);
                    self.state = State::Expr;
                }
                (State::Expr, '}') => {
                    if inexpr_open_count == 0 {
                        self.output.push(chr);
                        self.extracted_expressions.push(current_expr.trim().into());
                        current_expr = String::new();
                        self.state = State::NotExpr;
                    } else {
                        // We're closing one brace met before inside of the expression.
                        current_expr.push(chr);
                        inexpr_open_count -= 1;
                    }
                }
                (State::Expr, ':') => {
                    if inexpr_open_count == 0 {
                        // We're outside of braces, thus assume that it's a specifier, like "{Some(value):?}"
                        self.output.push(chr);
                        self.extracted_expressions.push(current_expr.trim().into());
                        current_expr = String::new();
                        self.state = State::FormatOpts;
                    } else {
                        // We're inside of braced expression, assume that it's a struct field name/value delimeter.
                        current_expr.push(chr);
                    }
                }
                (State::Expr, '{') => {
                    current_expr.push(chr);
                    inexpr_open_count += 1;
                }
                (State::Expr, _) => {
                    current_expr.push(chr);
                }
                (State::FormatOpts, '}') => {
                    self.output.push(chr);
                    self.state = State::NotExpr;
                }
                (State::FormatOpts, _) => {
                    self.output.push(chr);
                }
            }
        }

        if self.state != State::NotExpr {
            return Err(());
        }

        self.parsed = true;
        Ok(())
    }

    pub fn into_suggestion(&self, kind: PostfixKind) -> String {
        assert!(self.parsed, "Attempt to get a suggestion from not parsed expression");

        let mut output = format!(r#"{}("{}""#, kind.into_macro_name(), self.output);
        for expr in &self.extracted_expressions {
            output += ", ";
            output += expr;
        }
        output.push(')');

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_str_parser() {
        let test_vector = &[
            ("no expressions", Some(("no expressions", vec![]))),
            ("{expr} is {2 + 2}", Some(("{} is {}", vec!["expr", "2 + 2"]))),
            ("{expr:?}", Some(("{:?}", vec!["expr"]))),
            ("{malformed", None),
            ("malformed}", None),
            ("{{correct", Some(("{{correct", vec![]))),
            ("correct}}", Some(("correct}}", vec![]))),
            ("{correct}}}", Some(("{}}}", vec!["correct"]))),
            ("{correct}}}}}", Some(("{}}}}}", vec!["correct"]))),
            ("{incorrect}}", None),
            ("placeholders {} {}", Some(("placeholders {} {}", vec!["$1", "$2"]))),
            ("mixed {} {2 + 2} {}", Some(("mixed {} {} {}", vec!["$1", "2 + 2", "$2"]))),
            (
                "{SomeStruct { val_a: 0, val_b: 1 }}",
                Some(("{}", vec!["SomeStruct { val_a: 0, val_b: 1 }"])),
            ),
            ("{expr:?} is {2.32f64:.5}", Some(("{:?} is {:.5}", vec!["expr", "2.32f64"]))),
            (
                "{SomeStruct { val_a: 0, val_b: 1 }:?}",
                Some(("{:?}", vec!["SomeStruct { val_a: 0, val_b: 1 }"])),
            ),
            ("{     2 + 2        }", Some(("{}", vec!["2 + 2"]))),
        ];

        for (input, output) in test_vector {
            let mut parser = FormatStrParser::new(*input);
            let outcome = parser.parse();

            if let Some((result_str, result_args)) = output {
                assert!(
                    outcome.is_ok(),
                    "Outcome is error for input: {}, but the expected outcome is {:?}",
                    input,
                    output
                );
                assert_eq!(parser.output, *result_str);
                assert_eq!(&parser.extracted_expressions, result_args);
            } else {
                assert!(
                    outcome.is_err(),
                    "Outcome is OK for input: {}, but the expected outcome is error",
                    input
                );
            }
        }
    }

    #[test]
    fn test_into_suggestion() {
        let test_vector = &[
            (PostfixKind::Println, "{}", r#"println!("{}", $1)"#),
            (
                PostfixKind::LogInfo,
                "{} {expr} {} {2 + 2}",
                r#"log::info!("{} {} {} {}", $1, expr, $2, 2 + 2)"#,
            ),
            (PostfixKind::Format, "{expr:?}", r#"format!("{:?}", expr)"#),
        ];

        for (kind, input, output) in test_vector {
            let mut parser = FormatStrParser::new(*input);
            parser.parse().expect("Parsing must succeed");

            assert_eq!(&parser.into_suggestion(*kind), output);
        }
    }
}
