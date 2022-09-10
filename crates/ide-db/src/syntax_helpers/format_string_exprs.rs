//! Tools to work with expressions present in format string literals for the `format_args!` family of macros.
//! Primarily meant for assists and completions.

/// Enum for represenging extraced format string args.
/// Can either be extracted expressions (which includes identifiers),
/// or placeholders `{}`.
#[derive(Debug)]
pub enum Arg {
    Placeholder,
    Expr(String),
}

/**
 Add placeholders like `$1` and `$2` in place of [`Arg::Placeholder`].
 ```rust
 assert_eq!(vec![Arg::Expr("expr"), Arg::Placeholder, Arg::Expr("expr")], vec!["expr", "$1", "expr"])
 ```
*/

pub fn with_placeholders(args: Vec<Arg>) -> Vec<String> {
    let mut placeholder_id = 1;
    args.into_iter()
        .map(move |a| match a {
            Arg::Expr(s) => s,
            Arg::Placeholder => {
                let s = format!("${placeholder_id}");
                placeholder_id += 1;
                s
            }
        })
        .collect()
}

/**
 Parser for a format-like string. It is more allowing in terms of string contents,
 as we expect variable placeholders to be filled with expressions.

 Built for completions and assists, and escapes `\` and `$` in output.
 (See the comments on `get_receiver_text()` for detail.)
 Splits a format string that may contain expressions
 like
 ```rust
 assert_eq!(parse("{expr} {} {expr} ").unwrap(), ("{} {} {}", vec![Arg::Expr("expr"), Arg::Placeholder, Arg::Expr("expr")]));
 ```
*/
pub fn parse_format_exprs(input: &str) -> Result<(String, Vec<Arg>), ()> {
    #[derive(Debug, Clone, Copy, PartialEq)]
    enum State {
        NotExpr,
        MaybeExpr,
        Expr,
        MaybeIncorrect,
        FormatOpts,
    }

    let mut current_expr = String::new();
    let mut state = State::NotExpr;
    let mut extracted_expressions = Vec::new();
    let mut output = String::new();

    // Count of open braces inside of an expression.
    // We assume that user knows what they're doing, thus we treat it like a correct pattern, e.g.
    // "{MyStruct { val_a: 0, val_b: 1 }}".
    let mut inexpr_open_count = 0;

    let mut chars = input.chars().peekable();
    while let Some(chr) = chars.next() {
        match (state, chr) {
            (State::NotExpr, '{') => {
                output.push(chr);
                state = State::MaybeExpr;
            }
            (State::NotExpr, '}') => {
                output.push(chr);
                state = State::MaybeIncorrect;
            }
            (State::NotExpr, _) => {
                if matches!(chr, '\\' | '$') {
                    output.push('\\');
                }
                output.push(chr);
            }
            (State::MaybeIncorrect, '}') => {
                // It's okay, we met "}}".
                output.push(chr);
                state = State::NotExpr;
            }
            (State::MaybeIncorrect, _) => {
                // Error in the string.
                return Err(());
            }
            (State::MaybeExpr, '{') => {
                output.push(chr);
                state = State::NotExpr;
            }
            (State::MaybeExpr, '}') => {
                // This is an empty sequence '{}'. Replace it with placeholder.
                output.push(chr);
                extracted_expressions.push(Arg::Placeholder);
                state = State::NotExpr;
            }
            (State::MaybeExpr, _) => {
                if matches!(chr, '\\' | '$') {
                    current_expr.push('\\');
                }
                current_expr.push(chr);
                state = State::Expr;
            }
            (State::Expr, '}') => {
                if inexpr_open_count == 0 {
                    output.push(chr);
                    extracted_expressions.push(Arg::Expr(current_expr.trim().into()));
                    current_expr = String::new();
                    state = State::NotExpr;
                } else {
                    // We're closing one brace met before inside of the expression.
                    current_expr.push(chr);
                    inexpr_open_count -= 1;
                }
            }
            (State::Expr, ':') if matches!(chars.peek(), Some(':')) => {
                // path separator
                current_expr.push_str("::");
                chars.next();
            }
            (State::Expr, ':') => {
                if inexpr_open_count == 0 {
                    // We're outside of braces, thus assume that it's a specifier, like "{Some(value):?}"
                    output.push(chr);
                    extracted_expressions.push(Arg::Expr(current_expr.trim().into()));
                    current_expr = String::new();
                    state = State::FormatOpts;
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
                output.push(chr);
                state = State::NotExpr;
            }
            (State::FormatOpts, _) => {
                if matches!(chr, '\\' | '$') {
                    output.push('\\');
                }
                output.push(chr);
            }
        }
    }

    if state != State::NotExpr {
        return Err(());
    }

    Ok((output, extracted_expressions))
}

#[cfg(test)]
mod tests {
    use super::*;
    use expect_test::{expect, Expect};

    fn check(input: &str, expect: &Expect) {
        let (output, exprs) = parse_format_exprs(input).unwrap_or(("-".to_string(), vec![]));
        let outcome_repr = if !exprs.is_empty() {
            format!("{}; {}", output, with_placeholders(exprs).join(", "))
        } else {
            output
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
}
