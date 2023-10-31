//! Tools to work with expressions present in format string literals for the `format_args!` family of macros.
//! Primarily meant for assists and completions.

/// Enum for representing extracted format string args.
/// Can either be extracted expressions (which includes identifiers),
/// or placeholders `{}`.
#[derive(Debug, PartialEq, Eq)]
pub enum Arg {
    Placeholder,
    Ident(String),
    Expr(String),
}

/**
 Add placeholders like `$1` and `$2` in place of [`Arg::Placeholder`],
 and unwraps the [`Arg::Ident`] and [`Arg::Expr`] enums.
 ```rust
 # use ide_db::syntax_helpers::format_string_exprs::*;
 assert_eq!(with_placeholders(vec![Arg::Ident("ident".to_owned()), Arg::Placeholder, Arg::Expr("expr + 2".to_owned())]), vec!["ident".to_owned(), "$1".to_owned(), "expr + 2".to_owned()])
 ```
*/

pub fn with_placeholders(args: Vec<Arg>) -> Vec<String> {
    let mut placeholder_id = 1;
    args.into_iter()
        .map(move |a| match a {
            Arg::Expr(s) | Arg::Ident(s) => s,
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
 assert_eq!(parse("{ident} {} {expr + 42} ").unwrap(), ("{} {} {}", vec![Arg::Ident("ident"), Arg::Placeholder, Arg::Expr("expr + 42")]));
 ```
*/
pub fn parse_format_exprs(input: &str) -> Result<(String, Vec<Arg>), ()> {
    #[derive(Debug, Clone, Copy, PartialEq)]
    enum State {
        NotArg,
        MaybeArg,
        Expr,
        Ident,
        MaybeIncorrect,
        FormatOpts,
    }

    let mut state = State::NotArg;
    let mut current_expr = String::new();
    let mut extracted_expressions = Vec::new();
    let mut output = String::new();

    // Count of open braces inside of an expression.
    // We assume that user knows what they're doing, thus we treat it like a correct pattern, e.g.
    // "{MyStruct { val_a: 0, val_b: 1 }}".
    let mut inexpr_open_count = 0;

    let mut chars = input.chars().peekable();
    while let Some(chr) = chars.next() {
        match (state, chr) {
            (State::NotArg, '{') => {
                output.push(chr);
                state = State::MaybeArg;
            }
            (State::NotArg, '}') => {
                output.push(chr);
                state = State::MaybeIncorrect;
            }
            (State::NotArg, _) => {
                if matches!(chr, '\\' | '$') {
                    output.push('\\');
                }
                output.push(chr);
            }
            (State::MaybeIncorrect, '}') => {
                // It's okay, we met "}}".
                output.push(chr);
                state = State::NotArg;
            }
            (State::MaybeIncorrect, _) => {
                // Error in the string.
                return Err(());
            }
            // Escaped braces `{{`
            (State::MaybeArg, '{') => {
                output.push(chr);
                state = State::NotArg;
            }
            (State::MaybeArg, '}') => {
                // This is an empty sequence '{}'.
                output.push(chr);
                extracted_expressions.push(Arg::Placeholder);
                state = State::NotArg;
            }
            (State::MaybeArg, ':') => {
                output.push(chr);
                extracted_expressions.push(Arg::Placeholder);
                state = State::FormatOpts;
            }
            (State::MaybeArg, _) => {
                if matches!(chr, '\\' | '$') {
                    current_expr.push('\\');
                }
                current_expr.push(chr);

                // While Rust uses the unicode sets of XID_start and XID_continue for Identifiers
                // this is probably the best we can do to avoid a false positive
                if chr.is_alphabetic() || chr == '_' {
                    state = State::Ident;
                } else {
                    state = State::Expr;
                }
            }
            (State::Ident | State::Expr, ':') if matches!(chars.peek(), Some(':')) => {
                // path separator
                state = State::Expr;
                current_expr.push_str("::");
                chars.next();
            }
            (State::Ident | State::Expr, ':' | '}') => {
                if inexpr_open_count == 0 {
                    let trimmed = current_expr.trim();

                    // if the expression consists of a single number, like "0" or "12", it can refer to
                    // format args in the order they are specified.
                    // see: https://doc.rust-lang.org/std/fmt/#positional-parameters
                    if trimmed.chars().fold(true, |only_num, c| c.is_ascii_digit() && only_num) {
                        output.push_str(trimmed);
                    } else if matches!(state, State::Expr) {
                        extracted_expressions.push(Arg::Expr(trimmed.into()));
                    } else if matches!(state, State::Ident) {
                        output.push_str(trimmed);
                    }

                    output.push(chr);
                    current_expr.clear();
                    state = if chr == ':' {
                        State::FormatOpts
                    } else if chr == '}' {
                        State::NotArg
                    } else {
                        unreachable!()
                    };
                } else if chr == '}' {
                    // We're closing one brace met before inside of the expression.
                    current_expr.push(chr);
                    inexpr_open_count -= 1;
                } else if chr == ':' {
                    // We're inside of braced expression, assume that it's a struct field name/value delimiter.
                    current_expr.push(chr);
                }
            }
            (State::Ident | State::Expr, '{') => {
                state = State::Expr;
                current_expr.push(chr);
                inexpr_open_count += 1;
            }
            (State::Ident | State::Expr, _) => {
                if !(chr.is_alphanumeric() || chr == '_' || chr == '#') {
                    state = State::Expr;
                }

                if matches!(chr, '\\' | '$') {
                    current_expr.push('\\');
                }
                current_expr.push(chr);
            }
            (State::FormatOpts, '}') => {
                output.push(chr);
                state = State::NotArg;
            }
            (State::FormatOpts, _) => {
                if matches!(chr, '\\' | '$') {
                    output.push('\\');
                }
                output.push(chr);
            }
        }
    }

    if state != State::NotArg {
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
            format!("{output}; {}", with_placeholders(exprs).join(", "))
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
            ("{expr} is {2 + 2}", expect![["{expr} is {}; 2 + 2"]]),
            ("{expr:?}", expect![["{expr:?}"]]),
            ("{expr:1$}", expect![[r"{expr:1\$}"]]),
            ("{:1$}", expect![[r"{:1\$}; $1"]]),
            ("{:>padding$}", expect![[r"{:>padding\$}; $1"]]),
            ("{}, {}, {0}", expect![[r"{}, {}, {0}; $1, $2"]]),
            ("{}, {}, {0:b}", expect![[r"{}, {}, {0:b}; $1, $2"]]),
            ("{$0}", expect![[r"{}; \$0"]]),
            ("{malformed", expect![["-"]]),
            ("malformed}", expect![["-"]]),
            ("{{correct", expect![["{{correct"]]),
            ("correct}}", expect![["correct}}"]]),
            ("{correct}}}", expect![["{correct}}}"]]),
            ("{correct}}}}}", expect![["{correct}}}}}"]]),
            ("{incorrect}}", expect![["-"]]),
            ("placeholders {} {}", expect![["placeholders {} {}; $1, $2"]]),
            ("mixed {} {2 + 2} {}", expect![["mixed {} {} {}; $1, 2 + 2, $2"]]),
            (
                "{SomeStruct { val_a: 0, val_b: 1 }}",
                expect![["{}; SomeStruct { val_a: 0, val_b: 1 }"]],
            ),
            ("{expr:?} is {2.32f64:.5}", expect![["{expr:?} is {:.5}; 2.32f64"]]),
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
    fn arg_type() {
        assert_eq!(
            parse_format_exprs("{_ident} {r#raw_ident} {expr.obj} {name {thing: 42} } {}")
                .unwrap()
                .1,
            vec![
                Arg::Expr("expr.obj".to_owned()),
                Arg::Expr("name {thing: 42}".to_owned()),
                Arg::Placeholder
            ]
        );
    }
}
