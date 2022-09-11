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
// image::https://user-images.githubusercontent.com/48062697/113020656-b560f500-917a-11eb-87de-02991f61beb8.gif[]

use ide_db::SnippetCap;
use syntax::ast::{self, AstToken};

use crate::{
    completions::postfix::build_postfix_snippet_builder, context::CompletionContext, Completions,
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
    let input = match string_literal_contents(receiver_text) {
        // It's not a string literal, do not parse input.
        Some(input) => input,
        None => return,
    };

    let postfix_snippet = match build_postfix_snippet_builder(ctx, cap, dot_receiver) {
        Some(it) => it,
        None => return,
    };
    let mut parser = FormatStrParser::new(input);

    if parser.parse().is_ok() {
        for (label, macro_name) in KINDS {
            let snippet = parser.to_suggestion(macro_name);

            postfix_snippet(label, macro_name, &snippet).add_to(acc);
        }
    }
}

/// Checks whether provided item is a string literal.
fn string_literal_contents(item: &ast::String) -> Option<String> {
    let item = item.text();
    if item.len() >= 2 && item.starts_with('\"') && item.ends_with('\"') {
        return Some(item[1..item.len() - 1].to_owned());
    }

    None
}

/// Parser for a format-like string. It is more allowing in terms of string contents,
/// as we expect variable placeholders to be filled with expressions.
#[derive(Debug)]
pub(crate) struct FormatStrParser {
    input: String,
    output: String,
    extracted_expressions: Vec<String>,
    state: State,
    parsed: bool,
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
    pub(crate) fn new(input: String) -> Self {
        Self {
            input,
            output: String::new(),
            extracted_expressions: Vec::new(),
            state: State::NotExpr,
            parsed: false,
        }
    }

    pub(crate) fn parse(&mut self) -> Result<(), ()> {
        let mut current_expr = String::new();

        let mut placeholder_id = 1;

        // Count of open braces inside of an expression.
        // We assume that user knows what they're doing, thus we treat it like a correct pattern, e.g.
        // "{MyStruct { val_a: 0, val_b: 1 }}".
        let mut inexpr_open_count = 0;

        // We need to escape '\' and '$'. See the comments on `get_receiver_text()` for detail.
        let mut chars = self.input.chars().peekable();
        while let Some(chr) = chars.next() {
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
                    if matches!(chr, '\\' | '$') {
                        self.output.push('\\');
                    }
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
                    if matches!(chr, '\\' | '$') {
                        current_expr.push('\\');
                    }
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
                (State::Expr, ':') if chars.peek().copied() == Some(':') => {
                    // path separator
                    current_expr.push_str("::");
                    chars.next();
                }
                (State::Expr, ':') => {
                    if inexpr_open_count == 0 {
                        // We're outside of braces, thus assume that it's a specifier, like "{Some(value):?}"
                        self.output.push(chr);
                        self.extracted_expressions.push(current_expr.trim().into());
                        current_expr = String::new();
                        self.state = State::FormatOpts;
                    } else {
                        // We're inside of braced expression, assume that it's a struct field name/value delimiter.
                        current_expr.push(chr);
                    }
                }
                (State::Expr, '{') => {
                    current_expr.push(chr);
                    inexpr_open_count += 1;
                }
                (State::Expr, _) => {
                    if matches!(chr, '\\' | '$') {
                        current_expr.push('\\');
                    }
                    current_expr.push(chr);
                }
                (State::FormatOpts, '}') => {
                    self.output.push(chr);
                    self.state = State::NotExpr;
                }
                (State::FormatOpts, _) => {
                    if matches!(chr, '\\' | '$') {
                        self.output.push('\\');
                    }
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

    pub(crate) fn to_suggestion(&self, macro_name: &str) -> String {
        assert!(self.parsed, "Attempt to get a suggestion from not parsed expression");

        let expressions_as_string = self.extracted_expressions.join(", ");
        format!(r#"{}("{}", {})"#, macro_name, self.output, expressions_as_string)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use expect_test::{expect, Expect};

    fn check(input: &str, expect: &Expect) {
        let mut parser = FormatStrParser::new((*input).to_owned());
        let outcome_repr = if parser.parse().is_ok() {
            // Parsing should be OK, expected repr is "string; expr_1, expr_2".
            if parser.extracted_expressions.is_empty() {
                parser.output
            } else {
                format!("{}; {}", parser.output, parser.extracted_expressions.join(", "))
            }
        } else {
            // Parsing should fail, expected repr is "-".
            "-".to_owned()
        };

        expect.assert_eq(&outcome_repr);
    }

    #[test]
    fn format_str_parser() {
        let test_vector = &[
            ("no expressions", expect![["no expressions"]]),
            (r"no expressions with \$0$1", expect![r"no expressions with \\\$0\$1"]),
            ("{expr} is {2 + 2}", expect![["{} is {}; expr, 2 + 2"]]),
            ("{expr:?}", expect![["{:?}; expr"]]),
            ("{expr:1$}", expect![[r"{:1\$}; expr"]]),
            ("{$0}", expect![[r"{}; \$0"]]),
            ("{malformed", expect![["-"]]),
            ("malformed}", expect![["-"]]),
            ("{{correct", expect![["{{correct"]]),
            ("correct}}", expect![["correct}}"]]),
            ("{correct}}}", expect![["{}}}; correct"]]),
            ("{correct}}}}}", expect![["{}}}}}; correct"]]),
            ("{incorrect}}", expect![["-"]]),
            ("placeholders {} {}", expect![["placeholders {} {}; $1, $2"]]),
            ("mixed {} {2 + 2} {}", expect![["mixed {} {} {}; $1, 2 + 2, $2"]]),
            (
                "{SomeStruct { val_a: 0, val_b: 1 }}",
                expect![["{}; SomeStruct { val_a: 0, val_b: 1 }"]],
            ),
            ("{expr:?} is {2.32f64:.5}", expect![["{:?} is {:.5}; expr, 2.32f64"]]),
            (
                "{SomeStruct { val_a: 0, val_b: 1 }:?}",
                expect![["{:?}; SomeStruct { val_a: 0, val_b: 1 }"]],
            ),
            ("{     2 + 2        }", expect![["{}; 2 + 2"]]),
            ("{strsim::jaro_winkle(a)}", expect![["{}; strsim::jaro_winkle(a)"]]),
            ("{foo::bar::baz()}", expect![["{}; foo::bar::baz()"]]),
            ("{foo::bar():?}", expect![["{:?}; foo::bar()"]]),
        ];

        for (input, output) in test_vector {
            check(input, output)
        }
    }

    #[test]
    fn test_into_suggestion() {
        let test_vector = &[
            ("println!", "{}", r#"println!("{}", $1)"#),
            ("eprintln!", "{}", r#"eprintln!("{}", $1)"#),
            (
                "log::info!",
                "{} {expr} {} {2 + 2}",
                r#"log::info!("{} {} {} {}", $1, expr, $2, 2 + 2)"#,
            ),
            ("format!", "{expr:?}", r#"format!("{:?}", expr)"#),
        ];

        for (kind, input, output) in test_vector {
            let mut parser = FormatStrParser::new((*input).to_owned());
            parser.parse().expect("Parsing must succeed");

            assert_eq!(&parser.to_suggestion(*kind), output);
        }
    }
}
