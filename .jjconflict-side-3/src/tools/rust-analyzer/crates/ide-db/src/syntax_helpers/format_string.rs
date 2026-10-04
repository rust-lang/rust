//! Tools to work with format string literals for the `format_args!` family of macros.
use syntax::{
    AstNode, AstToken, TextRange, TextSize,
    ast::{self, IsString},
};

// FIXME: This can probably be re-implemented via the HIR?
pub fn is_format_string(string: &ast::String) -> bool {
    // Check if `string` is a format string argument of a macro invocation.
    // `string` is a string literal, mapped down into the innermost macro expansion.
    // Since `format_args!` etc. remove the format string when expanding, but place all arguments
    // in the expanded output, we know that the string token is (part of) the format string if it
    // appears in `format_args!` (otherwise it would have been mapped down further).
    //
    // This setup lets us correctly highlight the components of `concat!("{}", "bla")` format
    // strings. It still fails for `concat!("{", "}")`, but that is rare.
    (|| {
        let lit = string.syntax().parent().and_then(ast::Literal::cast)?;
        let fa = lit.syntax().parent().and_then(ast::FormatArgsExpr::cast)?;
        (fa.template()? == ast::Expr::Literal(lit)).then_some(|| ())
    })()
    .is_some()
}

#[derive(Debug)]
pub enum FormatSpecifier {
    Open,
    Close,
    Integer,
    Identifier,
    Colon,
    Fill,
    Align,
    Sign,
    NumberSign,
    Zero,
    DollarSign,
    Dot,
    Asterisk,
    QuestionMark,
    Escape,
}

// FIXME: Remove this, we can use rustc_format_parse instead
pub fn lex_format_specifiers(
    string: &ast::String,
    mut callback: &mut dyn FnMut(TextRange, FormatSpecifier),
) {
    let mut char_ranges = Vec::new();
    string.escaped_char_ranges(&mut |range, res| char_ranges.push((range, res)));
    let mut chars = char_ranges
        .iter()
        .filter_map(|(range, res)| Some((*range, *res.as_ref().ok()?)))
        .peekable();

    while let Some((range, first_char)) = chars.next() {
        if let '{' = first_char {
            // Format specifier, see syntax at https://doc.rust-lang.org/std/fmt/index.html#syntax
            if let Some((_, '{')) = chars.peek() {
                // Escaped format specifier, `{{`
                read_escaped_format_specifier(&mut chars, &mut callback);
                continue;
            }

            callback(range, FormatSpecifier::Open);

            // check for integer/identifier
            let (_, int_char) = chars.peek().copied().unwrap_or_default();
            match int_char {
                // integer
                '0'..='9' => read_integer(&mut chars, &mut callback),
                // identifier
                c if c == '_' || c.is_alphabetic() => read_identifier(&mut chars, &mut callback),
                _ => {}
            }

            if let Some((_, ':')) = chars.peek() {
                skip_char_and_emit(&mut chars, FormatSpecifier::Colon, &mut callback);

                // check for fill/align
                let mut cloned = chars.clone().take(2);
                let (_, first) = cloned.next().unwrap_or_default();
                let (_, second) = cloned.next().unwrap_or_default();
                match second {
                    '<' | '^' | '>' => {
                        // alignment specifier, first char specifies fill
                        skip_char_and_emit(&mut chars, FormatSpecifier::Fill, &mut callback);
                        skip_char_and_emit(&mut chars, FormatSpecifier::Align, &mut callback);
                    }
                    _ => {
                        if let '<' | '^' | '>' = first {
                            skip_char_and_emit(&mut chars, FormatSpecifier::Align, &mut callback);
                        }
                    }
                }

                // check for sign
                match chars.peek().copied().unwrap_or_default().1 {
                    '+' | '-' => {
                        skip_char_and_emit(&mut chars, FormatSpecifier::Sign, &mut callback);
                    }
                    _ => {}
                }

                // check for `#`
                if let Some((_, '#')) = chars.peek() {
                    skip_char_and_emit(&mut chars, FormatSpecifier::NumberSign, &mut callback);
                }

                // check for `0`
                let mut cloned = chars.clone().take(2);
                let first = cloned.next().map(|next| next.1);
                let second = cloned.next().map(|next| next.1);

                if first == Some('0') && second != Some('$') {
                    skip_char_and_emit(&mut chars, FormatSpecifier::Zero, &mut callback);
                }

                // width
                match chars.peek().copied().unwrap_or_default().1 {
                    '0'..='9' => {
                        read_integer(&mut chars, &mut callback);
                        if let Some((_, '$')) = chars.peek() {
                            skip_char_and_emit(
                                &mut chars,
                                FormatSpecifier::DollarSign,
                                &mut callback,
                            );
                        }
                    }
                    c if c == '_' || c.is_alphabetic() => {
                        read_identifier(&mut chars, &mut callback);

                        if chars.peek().map(|&(_, c)| c) == Some('?') {
                            skip_char_and_emit(
                                &mut chars,
                                FormatSpecifier::QuestionMark,
                                &mut callback,
                            );
                        }

                        // can be either width (indicated by dollar sign, or type in which case
                        // the next sign has to be `}`)
                        let next = chars.peek().map(|&(_, c)| c);

                        match next {
                            Some('$') => skip_char_and_emit(
                                &mut chars,
                                FormatSpecifier::DollarSign,
                                &mut callback,
                            ),
                            Some('}') => {
                                skip_char_and_emit(
                                    &mut chars,
                                    FormatSpecifier::Close,
                                    &mut callback,
                                );
                                continue;
                            }
                            _ => continue,
                        };
                    }
                    _ => {}
                }

                // precision
                if let Some((_, '.')) = chars.peek() {
                    skip_char_and_emit(&mut chars, FormatSpecifier::Dot, &mut callback);

                    match chars.peek().copied().unwrap_or_default().1 {
                        '*' => {
                            skip_char_and_emit(
                                &mut chars,
                                FormatSpecifier::Asterisk,
                                &mut callback,
                            );
                        }
                        '0'..='9' => {
                            read_integer(&mut chars, &mut callback);
                            if let Some((_, '$')) = chars.peek() {
                                skip_char_and_emit(
                                    &mut chars,
                                    FormatSpecifier::DollarSign,
                                    &mut callback,
                                );
                            }
                        }
                        c if c == '_' || c.is_alphabetic() => {
                            read_identifier(&mut chars, &mut callback);
                            if chars.peek().map(|&(_, c)| c) != Some('$') {
                                continue;
                            }
                            skip_char_and_emit(
                                &mut chars,
                                FormatSpecifier::DollarSign,
                                &mut callback,
                            );
                        }
                        _ => {
                            continue;
                        }
                    }
                }

                // type
                match chars.peek().copied().unwrap_or_default().1 {
                    '?' => {
                        skip_char_and_emit(
                            &mut chars,
                            FormatSpecifier::QuestionMark,
                            &mut callback,
                        );
                    }
                    c if c == '_' || c.is_alphabetic() => {
                        read_identifier(&mut chars, &mut callback);

                        if chars.peek().map(|&(_, c)| c) == Some('?') {
                            skip_char_and_emit(
                                &mut chars,
                                FormatSpecifier::QuestionMark,
                                &mut callback,
                            );
                        }
                    }
                    _ => {}
                }
            }

            if let Some((_, '}')) = chars.peek() {
                skip_char_and_emit(&mut chars, FormatSpecifier::Close, &mut callback);
            }
            continue;
        } else if let '}' = first_char
            && let Some((_, '}')) = chars.peek()
        {
            // Escaped format specifier, `}}`
            read_escaped_format_specifier(&mut chars, &mut callback);
        }
    }

    fn skip_char_and_emit<I, F>(
        chars: &mut std::iter::Peekable<I>,
        emit: FormatSpecifier,
        callback: &mut F,
    ) where
        I: Iterator<Item = (TextRange, char)>,
        F: FnMut(TextRange, FormatSpecifier),
    {
        let (range, _) = chars.next().unwrap();
        callback(range, emit);
    }

    fn read_integer<I, F>(chars: &mut std::iter::Peekable<I>, callback: &mut F)
    where
        I: Iterator<Item = (TextRange, char)>,
        F: FnMut(TextRange, FormatSpecifier),
    {
        let (mut range, c) = chars.next().unwrap();
        assert!(c.is_ascii_digit());
        while let Some(&(r, next_char)) = chars.peek() {
            if next_char.is_ascii_digit() {
                chars.next();
                range = range.cover(r);
            } else {
                break;
            }
        }
        callback(range, FormatSpecifier::Integer);
    }

    fn read_identifier<I, F>(chars: &mut std::iter::Peekable<I>, callback: &mut F)
    where
        I: Iterator<Item = (TextRange, char)>,
        F: FnMut(TextRange, FormatSpecifier),
    {
        let (mut range, c) = chars.next().unwrap();
        assert!(c.is_alphabetic() || c == '_');
        while let Some(&(r, next_char)) = chars.peek() {
            if next_char == '_' || next_char.is_ascii_digit() || next_char.is_alphabetic() {
                chars.next();
                range = range.cover(r);
            } else {
                break;
            }
        }
        callback(range, FormatSpecifier::Identifier);
    }

    fn read_escaped_format_specifier<I, F>(chars: &mut std::iter::Peekable<I>, callback: &mut F)
    where
        I: Iterator<Item = (TextRange, char)>,
        F: FnMut(TextRange, FormatSpecifier),
    {
        let (range, _) = chars.peek().unwrap();
        let offset = TextSize::from(1);
        callback(TextRange::new(range.start() - offset, range.end()), FormatSpecifier::Escape);
        chars.next();
    }
}
