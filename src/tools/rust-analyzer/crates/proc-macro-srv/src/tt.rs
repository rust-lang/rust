use core::fmt;
use std::sync::Arc;

use intern::Symbol;
use proc_macro::{Delimiter, bridge};
use rustc_lexer::{DocStyle, LiteralKind};

pub type TokenTree<S> = bridge::TokenTree<TokenStream<S>, S, Symbol>;

/// Trait for allowing integration tests to parse tokenstreams with dynamic span ranges
pub trait SpanLike {
    fn derive_ranged(&self, range: std::ops::Range<usize>) -> Self;
}

impl SpanLike for crate::SpanId {
    fn derive_ranged(&self, _: std::ops::Range<usize>) -> Self {
        *self
    }
}

impl SpanLike for () {
    fn derive_ranged(&self, _: std::ops::Range<usize>) -> Self {
        *self
    }
}

impl SpanLike for crate::Span {
    fn derive_ranged(&self, range: std::ops::Range<usize>) -> Self {
        crate::Span {
            range: span::TextRange::new(
                span::TextSize::new(range.start as u32),
                span::TextSize::new(range.end as u32),
            ),
            anchor: self.anchor,
            ctx: self.ctx,
        }
    }
}

#[derive(Clone)]
pub struct TokenStream<S>(pub(crate) Arc<Vec<TokenTree<S>>>);

impl<S> Default for TokenStream<S> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<S> TokenStream<S> {
    pub fn new(tts: Vec<TokenTree<S>>) -> TokenStream<S> {
        TokenStream(Arc::new(tts))
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn get(&self, index: usize) -> Option<&TokenTree<S>> {
        self.0.get(index)
    }

    pub fn iter(&self) -> TokenStreamIter<'_, S> {
        TokenStreamIter::new(self)
    }

    pub fn chunks(&self, chunk_size: usize) -> core::slice::Chunks<'_, TokenTree<S>> {
        self.0.chunks(chunk_size)
    }

    pub fn from_str(s: &str, span: S) -> Result<Self, String>
    where
        S: SpanLike + Copy,
    {
        let mut groups = Vec::new();
        groups.push((proc_macro::Delimiter::None, 0..0, vec![]));
        let mut offset = 0;
        let mut tokens = rustc_lexer::tokenize(s, rustc_lexer::FrontmatterAllowed::No).peekable();
        while let Some(token) = tokens.next() {
            let range = offset..offset + token.len as usize;
            offset += token.len as usize;

            let mut is_joint = || {
                tokens.peek().is_some_and(|token| {
                    matches!(
                        token.kind,
                        rustc_lexer::TokenKind::RawLifetime
                            | rustc_lexer::TokenKind::GuardedStrPrefix
                            | rustc_lexer::TokenKind::Lifetime { .. }
                            | rustc_lexer::TokenKind::Semi
                            | rustc_lexer::TokenKind::Comma
                            | rustc_lexer::TokenKind::Dot
                            | rustc_lexer::TokenKind::OpenParen
                            | rustc_lexer::TokenKind::CloseParen
                            | rustc_lexer::TokenKind::OpenBrace
                            | rustc_lexer::TokenKind::CloseBrace
                            | rustc_lexer::TokenKind::OpenBracket
                            | rustc_lexer::TokenKind::CloseBracket
                            | rustc_lexer::TokenKind::At
                            | rustc_lexer::TokenKind::Pound
                            | rustc_lexer::TokenKind::Tilde
                            | rustc_lexer::TokenKind::Question
                            | rustc_lexer::TokenKind::Colon
                            | rustc_lexer::TokenKind::Dollar
                            | rustc_lexer::TokenKind::Eq
                            | rustc_lexer::TokenKind::Bang
                            | rustc_lexer::TokenKind::Lt
                            | rustc_lexer::TokenKind::Gt
                            | rustc_lexer::TokenKind::Minus
                            | rustc_lexer::TokenKind::And
                            | rustc_lexer::TokenKind::Or
                            | rustc_lexer::TokenKind::Plus
                            | rustc_lexer::TokenKind::Star
                            | rustc_lexer::TokenKind::Slash
                            | rustc_lexer::TokenKind::Percent
                            | rustc_lexer::TokenKind::Caret
                    )
                })
            };

            let Some((open_delim, _, tokenstream)) = groups.last_mut() else {
                return Err("Unbalanced delimiters".to_owned());
            };
            match token.kind {
                rustc_lexer::TokenKind::OpenParen => {
                    groups.push((proc_macro::Delimiter::Parenthesis, range, vec![]))
                }
                rustc_lexer::TokenKind::CloseParen if *open_delim != Delimiter::Parenthesis => {
                    return Err("Expected ')'".to_owned());
                }
                rustc_lexer::TokenKind::CloseParen => {
                    let (delimiter, open_range, stream) = groups.pop().unwrap();
                    groups.last_mut().ok_or_else(|| "Unbalanced delimiters".to_owned())?.2.push(
                        TokenTree::Group(bridge::Group {
                            delimiter,
                            stream: if stream.is_empty() {
                                None
                            } else {
                                Some(TokenStream::new(stream))
                            },
                            span: bridge::DelimSpan {
                                entire: span.derive_ranged(open_range.start..range.end),
                                open: span.derive_ranged(open_range),
                                close: span.derive_ranged(range),
                            },
                        }),
                    );
                }
                rustc_lexer::TokenKind::OpenBrace => {
                    groups.push((proc_macro::Delimiter::Brace, range, vec![]))
                }
                rustc_lexer::TokenKind::CloseBrace if *open_delim != Delimiter::Brace => {
                    return Err("Expected '}'".to_owned());
                }
                rustc_lexer::TokenKind::CloseBrace => {
                    let (delimiter, open_range, stream) = groups.pop().unwrap();
                    groups.last_mut().ok_or_else(|| "Unbalanced delimiters".to_owned())?.2.push(
                        TokenTree::Group(bridge::Group {
                            delimiter,
                            stream: if stream.is_empty() {
                                None
                            } else {
                                Some(TokenStream::new(stream))
                            },
                            span: bridge::DelimSpan {
                                entire: span.derive_ranged(open_range.start..range.end),
                                open: span.derive_ranged(open_range),
                                close: span.derive_ranged(range),
                            },
                        }),
                    );
                }
                rustc_lexer::TokenKind::OpenBracket => {
                    groups.push((proc_macro::Delimiter::Bracket, range, vec![]))
                }
                rustc_lexer::TokenKind::CloseBracket if *open_delim != Delimiter::Bracket => {
                    return Err("Expected ']'".to_owned());
                }
                rustc_lexer::TokenKind::CloseBracket => {
                    let (delimiter, open_range, stream) = groups.pop().unwrap();
                    groups.last_mut().ok_or_else(|| "Unbalanced delimiters".to_owned())?.2.push(
                        TokenTree::Group(bridge::Group {
                            delimiter,
                            stream: if stream.is_empty() {
                                None
                            } else {
                                Some(TokenStream::new(stream))
                            },
                            span: bridge::DelimSpan {
                                entire: span.derive_ranged(open_range.start..range.end),
                                open: span.derive_ranged(open_range),
                                close: span.derive_ranged(range),
                            },
                        }),
                    );
                }
                rustc_lexer::TokenKind::LineComment { doc_style: None }
                | rustc_lexer::TokenKind::BlockComment { doc_style: None, terminated: _ } => {
                    continue;
                }
                rustc_lexer::TokenKind::LineComment { doc_style: Some(doc_style) } => {
                    let text = &s[range.start + 2..range.end];
                    tokenstream.push(bridge::TokenTree::Punct(bridge::Punct {
                        ch: b'#',
                        joint: false,
                        span,
                    }));
                    if doc_style == DocStyle::Inner {
                        tokenstream.push(bridge::TokenTree::Punct(bridge::Punct {
                            ch: b'!',
                            joint: false,
                            span,
                        }));
                    }
                    tokenstream.push(bridge::TokenTree::Group(bridge::Group {
                        delimiter: Delimiter::Bracket,
                        stream: Some(TokenStream::new(vec![
                            bridge::TokenTree::Ident(bridge::Ident {
                                sym: Symbol::intern("doc"),
                                is_raw: false,
                                span,
                            }),
                            bridge::TokenTree::Punct(bridge::Punct {
                                ch: b'=',
                                joint: false,
                                span,
                            }),
                            bridge::TokenTree::Literal(bridge::Literal {
                                kind: bridge::LitKind::Str,
                                symbol: Symbol::intern(&text.escape_debug().to_string()),
                                suffix: None,
                                span: span.derive_ranged(range),
                            }),
                        ])),
                        span: bridge::DelimSpan { open: span, close: span, entire: span },
                    }));
                }
                rustc_lexer::TokenKind::BlockComment { doc_style: Some(doc_style), terminated } => {
                    let text =
                        &s[range.start + 2..if terminated { range.end - 2 } else { range.end }];
                    tokenstream.push(bridge::TokenTree::Punct(bridge::Punct {
                        ch: b'#',
                        joint: false,
                        span,
                    }));
                    if doc_style == DocStyle::Inner {
                        tokenstream.push(bridge::TokenTree::Punct(bridge::Punct {
                            ch: b'!',
                            joint: false,
                            span,
                        }));
                    }
                    tokenstream.push(bridge::TokenTree::Group(bridge::Group {
                        delimiter: Delimiter::Bracket,
                        stream: Some(TokenStream::new(vec![
                            bridge::TokenTree::Ident(bridge::Ident {
                                sym: Symbol::intern("doc"),
                                is_raw: false,
                                span,
                            }),
                            bridge::TokenTree::Punct(bridge::Punct {
                                ch: b'=',
                                joint: false,
                                span,
                            }),
                            bridge::TokenTree::Literal(bridge::Literal {
                                kind: bridge::LitKind::Str,
                                symbol: Symbol::intern(&text.escape_debug().to_string()),
                                suffix: None,
                                span: span.derive_ranged(range),
                            }),
                        ])),
                        span: bridge::DelimSpan { open: span, close: span, entire: span },
                    }));
                }
                rustc_lexer::TokenKind::Whitespace => continue,
                rustc_lexer::TokenKind::Frontmatter { .. } => unreachable!(),
                rustc_lexer::TokenKind::Unknown => return Err("Unknown token".to_owned()),
                rustc_lexer::TokenKind::UnknownPrefix => return Err("Unknown prefix".to_owned()),
                rustc_lexer::TokenKind::UnknownPrefixLifetime => {
                    return Err("Unknown lifetime prefix".to_owned());
                }
                // FIXME: Error on edition >= 2024 ... I dont think the proc-macro server can fetch editions currently
                // and whose edition is this?
                rustc_lexer::TokenKind::GuardedStrPrefix => {
                    tokenstream.push(bridge::TokenTree::Punct(bridge::Punct {
                        ch: s.as_bytes()[range.start],
                        joint: true,
                        span: span.derive_ranged(range.start..range.start + 1),
                    }));
                    tokenstream.push(bridge::TokenTree::Punct(bridge::Punct {
                        ch: s.as_bytes()[range.start + 1],
                        joint: is_joint(),
                        span: span.derive_ranged(range.start + 1..range.end),
                    }))
                }
                rustc_lexer::TokenKind::Ident => {
                    tokenstream.push(bridge::TokenTree::Ident(bridge::Ident {
                        sym: Symbol::intern(&s[range.clone()]),
                        is_raw: false,
                        span: span.derive_ranged(range),
                    }))
                }
                rustc_lexer::TokenKind::InvalidIdent => return Err("Invalid identifier".to_owned()),
                rustc_lexer::TokenKind::RawIdent => {
                    let range = range.start + 2..range.end;
                    tokenstream.push(bridge::TokenTree::Ident(bridge::Ident {
                        sym: Symbol::intern(&s[range.clone()]),
                        is_raw: true,
                        span: span.derive_ranged(range),
                    }))
                }
                rustc_lexer::TokenKind::Literal { kind, suffix_start } => {
                    tokenstream.push(bridge::TokenTree::Literal(literal_from_lexer(
                        &s[range.clone()],
                        span.derive_ranged(range),
                        kind,
                        suffix_start,
                    )))
                }
                rustc_lexer::TokenKind::RawLifetime => {
                    let range = range.start + 1 + 2..range.end;
                    tokenstream.push(bridge::TokenTree::Punct(bridge::Punct {
                        ch: b'\'',
                        joint: true,
                        span: span.derive_ranged(range.start..range.start + 1),
                    }));
                    tokenstream.push(bridge::TokenTree::Ident(bridge::Ident {
                        sym: Symbol::intern(&s[range.clone()]),
                        is_raw: true,
                        span: span.derive_ranged(range),
                    }))
                }
                rustc_lexer::TokenKind::Lifetime { starts_with_number } => {
                    if starts_with_number {
                        return Err("Lifetime cannot start with a number".to_owned());
                    }
                    let range = range.start + 1..range.end;
                    tokenstream.push(bridge::TokenTree::Punct(bridge::Punct {
                        ch: b'\'',
                        joint: true,
                        span: span.derive_ranged(range.start..range.start + 1),
                    }));
                    tokenstream.push(bridge::TokenTree::Ident(bridge::Ident {
                        sym: Symbol::intern(&s[range.clone()]),
                        is_raw: false,
                        span: span.derive_ranged(range),
                    }))
                }
                rustc_lexer::TokenKind::Semi => {
                    tokenstream.push(bridge::TokenTree::Punct(bridge::Punct {
                        ch: b';',
                        joint: is_joint(),
                        span: span.derive_ranged(range),
                    }))
                }
                rustc_lexer::TokenKind::Comma => {
                    tokenstream.push(bridge::TokenTree::Punct(bridge::Punct {
                        ch: b',',
                        joint: is_joint(),
                        span: span.derive_ranged(range),
                    }))
                }
                rustc_lexer::TokenKind::Dot => {
                    tokenstream.push(bridge::TokenTree::Punct(bridge::Punct {
                        ch: b'.',
                        joint: is_joint(),
                        span: span.derive_ranged(range),
                    }))
                }
                rustc_lexer::TokenKind::At => {
                    tokenstream.push(bridge::TokenTree::Punct(bridge::Punct {
                        ch: b'@',
                        joint: is_joint(),
                        span: span.derive_ranged(range),
                    }))
                }
                rustc_lexer::TokenKind::Pound => {
                    tokenstream.push(bridge::TokenTree::Punct(bridge::Punct {
                        ch: b'#',
                        joint: is_joint(),
                        span: span.derive_ranged(range),
                    }))
                }
                rustc_lexer::TokenKind::Tilde => {
                    tokenstream.push(bridge::TokenTree::Punct(bridge::Punct {
                        ch: b'~',
                        joint: is_joint(),
                        span: span.derive_ranged(range),
                    }))
                }
                rustc_lexer::TokenKind::Question => {
                    tokenstream.push(bridge::TokenTree::Punct(bridge::Punct {
                        ch: b'?',
                        joint: is_joint(),
                        span: span.derive_ranged(range),
                    }))
                }
                rustc_lexer::TokenKind::Colon => {
                    tokenstream.push(bridge::TokenTree::Punct(bridge::Punct {
                        ch: b':',
                        joint: is_joint(),
                        span: span.derive_ranged(range),
                    }))
                }
                rustc_lexer::TokenKind::Dollar => {
                    tokenstream.push(bridge::TokenTree::Punct(bridge::Punct {
                        ch: b'$',
                        joint: is_joint(),
                        span: span.derive_ranged(range),
                    }))
                }
                rustc_lexer::TokenKind::Eq => {
                    tokenstream.push(bridge::TokenTree::Punct(bridge::Punct {
                        ch: b'=',
                        joint: is_joint(),
                        span: span.derive_ranged(range),
                    }))
                }
                rustc_lexer::TokenKind::Bang => {
                    tokenstream.push(bridge::TokenTree::Punct(bridge::Punct {
                        ch: b'!',
                        joint: is_joint(),
                        span: span.derive_ranged(range),
                    }))
                }
                rustc_lexer::TokenKind::Lt => {
                    tokenstream.push(bridge::TokenTree::Punct(bridge::Punct {
                        ch: b'<',
                        joint: is_joint(),
                        span: span.derive_ranged(range),
                    }))
                }
                rustc_lexer::TokenKind::Gt => {
                    tokenstream.push(bridge::TokenTree::Punct(bridge::Punct {
                        ch: b'>',
                        joint: is_joint(),
                        span: span.derive_ranged(range),
                    }))
                }
                rustc_lexer::TokenKind::Minus => {
                    tokenstream.push(bridge::TokenTree::Punct(bridge::Punct {
                        ch: b'-',
                        joint: is_joint(),
                        span: span.derive_ranged(range),
                    }))
                }
                rustc_lexer::TokenKind::And => {
                    tokenstream.push(bridge::TokenTree::Punct(bridge::Punct {
                        ch: b'&',
                        joint: is_joint(),
                        span: span.derive_ranged(range),
                    }))
                }
                rustc_lexer::TokenKind::Or => {
                    tokenstream.push(bridge::TokenTree::Punct(bridge::Punct {
                        ch: b'|',
                        joint: is_joint(),
                        span: span.derive_ranged(range),
                    }))
                }
                rustc_lexer::TokenKind::Plus => {
                    tokenstream.push(bridge::TokenTree::Punct(bridge::Punct {
                        ch: b'+',
                        joint: is_joint(),
                        span: span.derive_ranged(range),
                    }))
                }
                rustc_lexer::TokenKind::Star => {
                    tokenstream.push(bridge::TokenTree::Punct(bridge::Punct {
                        ch: b'*',
                        joint: is_joint(),
                        span: span.derive_ranged(range),
                    }))
                }
                rustc_lexer::TokenKind::Slash => {
                    tokenstream.push(bridge::TokenTree::Punct(bridge::Punct {
                        ch: b'/',
                        joint: is_joint(),
                        span: span.derive_ranged(range),
                    }))
                }
                rustc_lexer::TokenKind::Caret => {
                    tokenstream.push(bridge::TokenTree::Punct(bridge::Punct {
                        ch: b'^',
                        joint: is_joint(),
                        span: span.derive_ranged(range),
                    }))
                }
                rustc_lexer::TokenKind::Percent => {
                    tokenstream.push(bridge::TokenTree::Punct(bridge::Punct {
                        ch: b'%',
                        joint: is_joint(),
                        span: span.derive_ranged(range),
                    }))
                }
                rustc_lexer::TokenKind::Eof => break,
            }
        }
        if let Some((Delimiter::None, _, tokentrees)) = groups.pop()
            && groups.is_empty()
        {
            Ok(TokenStream::new(tokentrees))
        } else {
            Err("Mismatched token groups".to_owned())
        }
    }
}

impl<S> fmt::Display for TokenStream<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for tt in self.0.iter() {
            display_token_tree(tt, f)?;
        }
        Ok(())
    }
}

fn display_token_tree<S>(tt: &TokenTree<S>, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match tt {
        bridge::TokenTree::Group(bridge::Group { delimiter, stream, span: _ }) => {
            write!(
                f,
                "{}",
                match delimiter {
                    proc_macro::Delimiter::Parenthesis => "(",
                    proc_macro::Delimiter::Brace => "{",
                    proc_macro::Delimiter::Bracket => "[",
                    proc_macro::Delimiter::None => "",
                }
            )?;
            if let Some(stream) = stream {
                write!(f, "{stream}")?;
            }
            write!(
                f,
                "{}",
                match delimiter {
                    proc_macro::Delimiter::Parenthesis => ")",
                    proc_macro::Delimiter::Brace => "}",
                    proc_macro::Delimiter::Bracket => "]",
                    proc_macro::Delimiter::None => "",
                }
            )?;
        }
        bridge::TokenTree::Punct(bridge::Punct { ch, joint, span: _ }) => {
            write!(f, "{ch}{}", if *joint { "" } else { " " })?
        }
        bridge::TokenTree::Ident(bridge::Ident { sym, is_raw, span: _ }) => {
            if *is_raw {
                write!(f, "r#")?;
            }
            write!(f, "{sym} ")?;
        }
        bridge::TokenTree::Literal(lit) => {
            display_fmt_literal(lit, f)?;
            let joint = match lit.kind {
                bridge::LitKind::Str
                | bridge::LitKind::StrRaw(_)
                | bridge::LitKind::ByteStr
                | bridge::LitKind::ByteStrRaw(_)
                | bridge::LitKind::CStr
                | bridge::LitKind::CStrRaw(_) => true,
                _ => false,
            };
            if !joint {
                write!(f, " ")?;
            }
        }
    }
    Ok(())
}

fn display_fmt_literal<S>(
    literal: &bridge::Literal<S, Symbol>,
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    match literal.kind {
        bridge::LitKind::Byte => write!(f, "b'{}'", literal.symbol),
        bridge::LitKind::Char => write!(f, "'{}'", literal.symbol),
        bridge::LitKind::Integer | bridge::LitKind::Float | bridge::LitKind::ErrWithGuar => {
            write!(f, "{}", literal.symbol)
        }
        bridge::LitKind::Str => write!(f, "\"{}\"", literal.symbol),
        bridge::LitKind::ByteStr => write!(f, "b\"{}\"", literal.symbol),
        bridge::LitKind::CStr => write!(f, "c\"{}\"", literal.symbol),
        bridge::LitKind::StrRaw(num_of_hashes) => {
            let num_of_hashes = num_of_hashes as usize;
            write!(
                f,
                r#"r{0:#<num_of_hashes$}"{text}"{0:#<num_of_hashes$}"#,
                "",
                text = literal.symbol
            )
        }
        bridge::LitKind::ByteStrRaw(num_of_hashes) => {
            let num_of_hashes = num_of_hashes as usize;
            write!(
                f,
                r#"br{0:#<num_of_hashes$}"{text}"{0:#<num_of_hashes$}"#,
                "",
                text = literal.symbol
            )
        }
        bridge::LitKind::CStrRaw(num_of_hashes) => {
            let num_of_hashes = num_of_hashes as usize;
            write!(
                f,
                r#"cr{0:#<num_of_hashes$}"{text}"{0:#<num_of_hashes$}"#,
                "",
                text = literal.symbol
            )
        }
    }?;
    if let Some(suffix) = &literal.suffix {
        write!(f, "{suffix}")?;
    }
    Ok(())
}

impl<S: fmt::Debug> fmt::Debug for TokenStream<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        debug_token_stream(self, 0, f)
    }
}

fn debug_token_stream<S: fmt::Debug>(
    ts: &TokenStream<S>,
    depth: usize,
    f: &mut std::fmt::Formatter<'_>,
) -> std::fmt::Result {
    for tt in ts.0.iter() {
        debug_token_tree(tt, depth, f)?;
    }
    Ok(())
}

fn debug_token_tree<S: fmt::Debug>(
    tt: &TokenTree<S>,
    depth: usize,
    f: &mut std::fmt::Formatter<'_>,
) -> std::fmt::Result {
    write!(f, "{:indent$}", "", indent = depth * 2)?;
    match tt {
        bridge::TokenTree::Group(bridge::Group { delimiter, stream, span }) => {
            writeln!(
                f,
                "GROUP {}{} {:#?} {:#?} {:#?}",
                match delimiter {
                    proc_macro::Delimiter::Parenthesis => "(",
                    proc_macro::Delimiter::Brace => "{",
                    proc_macro::Delimiter::Bracket => "[",
                    proc_macro::Delimiter::None => "$",
                },
                match delimiter {
                    proc_macro::Delimiter::Parenthesis => ")",
                    proc_macro::Delimiter::Brace => "}",
                    proc_macro::Delimiter::Bracket => "]",
                    proc_macro::Delimiter::None => "$",
                },
                span.open,
                span.close,
                span.entire,
            )?;
            if let Some(stream) = stream {
                debug_token_stream(stream, depth + 1, f)?;
            }
            return Ok(());
        }
        bridge::TokenTree::Punct(bridge::Punct { ch, joint, span }) => write!(
            f,
            "PUNCT {span:#?} {} {}",
            *ch as char,
            if *joint { "[joint]" } else { "[alone]" }
        )?,
        bridge::TokenTree::Ident(bridge::Ident { sym, is_raw, span }) => {
            write!(f, "IDENT {span:#?} ")?;
            if *is_raw {
                write!(f, "r#")?;
            }
            write!(f, "{sym}")?;
        }
        bridge::TokenTree::Literal(bridge::Literal { kind, symbol, suffix, span }) => write!(
            f,
            "LITER {span:#?} {kind:?} {symbol}{} ",
            match suffix {
                Some(suffix) => suffix.clone(),
                None => Symbol::intern(""),
            }
        )?,
    }
    writeln!(f)
}

impl<S: Copy> TokenStream<S> {
    /// Push `tt` onto the end of the stream, possibly gluing it to the last
    /// token. Uses `make_mut` to maximize efficiency.
    pub fn push_tree(&mut self, tt: TokenTree<S>) {
        let vec_mut = Arc::make_mut(&mut self.0);
        vec_mut.push(tt);
    }

    /// Push `stream` onto the end of the stream, possibly gluing the first
    /// token tree to the last token. (No other token trees will be glued.)
    /// Uses `make_mut` to maximize efficiency.
    pub fn push_stream(&mut self, stream: TokenStream<S>) {
        let vec_mut = Arc::make_mut(&mut self.0);

        let stream_iter = stream.0.iter().cloned();

        vec_mut.extend(stream_iter);
    }
}

impl<S> FromIterator<TokenTree<S>> for TokenStream<S> {
    fn from_iter<I: IntoIterator<Item = TokenTree<S>>>(iter: I) -> Self {
        TokenStream::new(iter.into_iter().collect::<Vec<TokenTree<S>>>())
    }
}

#[derive(Clone)]
pub struct TokenStreamIter<'t, S> {
    stream: &'t TokenStream<S>,
    index: usize,
}

impl<'t, S> TokenStreamIter<'t, S> {
    fn new(stream: &'t TokenStream<S>) -> Self {
        TokenStreamIter { stream, index: 0 }
    }

    // Peeking could be done via `Peekable`, but most iterators need peeking,
    // and this is simple and avoids the need to use `peekable` and `Peekable`
    // at all the use sites.
    pub fn peek(&self) -> Option<&'t TokenTree<S>> {
        self.stream.0.get(self.index)
    }
}

impl<'t, S> Iterator for TokenStreamIter<'t, S> {
    type Item = &'t TokenTree<S>;

    fn next(&mut self) -> Option<&'t TokenTree<S>> {
        self.stream.0.get(self.index).map(|tree| {
            self.index += 1;
            tree
        })
    }
}

pub(super) fn literal_from_lexer<Span>(
    s: &str,
    span: Span,
    kind: rustc_lexer::LiteralKind,
    suffix_start: u32,
) -> bridge::Literal<Span, Symbol> {
    let (kind, start_offset, end_offset) = match kind {
        LiteralKind::Int { .. } => (bridge::LitKind::Integer, 0, 0),
        LiteralKind::Float { .. } => (bridge::LitKind::Float, 0, 0),
        LiteralKind::Char { terminated } => (bridge::LitKind::Char, 1, terminated as usize),
        LiteralKind::Byte { terminated } => (bridge::LitKind::Byte, 2, terminated as usize),
        LiteralKind::Str { terminated } => (bridge::LitKind::Str, 1, terminated as usize),
        LiteralKind::ByteStr { terminated } => (bridge::LitKind::ByteStr, 2, terminated as usize),
        LiteralKind::CStr { terminated } => (bridge::LitKind::CStr, 2, terminated as usize),
        LiteralKind::RawStr { n_hashes } => (
            bridge::LitKind::StrRaw(n_hashes.unwrap_or_default()),
            2 + n_hashes.unwrap_or_default() as usize,
            1 + n_hashes.unwrap_or_default() as usize,
        ),
        LiteralKind::RawByteStr { n_hashes } => (
            bridge::LitKind::ByteStrRaw(n_hashes.unwrap_or_default()),
            3 + n_hashes.unwrap_or_default() as usize,
            1 + n_hashes.unwrap_or_default() as usize,
        ),
        LiteralKind::RawCStr { n_hashes } => (
            bridge::LitKind::CStrRaw(n_hashes.unwrap_or_default()),
            3 + n_hashes.unwrap_or_default() as usize,
            1 + n_hashes.unwrap_or_default() as usize,
        ),
    };

    let (lit, suffix) = s.split_at(suffix_start as usize);
    let lit = &lit[start_offset..lit.len() - end_offset];
    let suffix = match suffix {
        "" | "_" => None,
        suffix => Some(Symbol::intern(suffix)),
    };

    bridge::Literal { kind, symbol: Symbol::intern(lit), suffix, span }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let token_stream = TokenStream::from_str("struct T {\"string\"}", ()).unwrap();
        token_stream.to_string();
        assert_eq!(token_stream.to_string(), "struct T {\"string\"}");
    }
}
