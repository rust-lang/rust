//! Convert macro-by-example tokens which are specific to macro expansion into a
//! format that works for our parser.

use std::fmt;
use std::hash::Hash;

use rustc_hash::FxHashMap;
use span::{Edition, SpanData};
use syntax::{SyntaxKind, SyntaxKind::*, T};

pub fn to_parser_input<Ctx: Copy + fmt::Debug + PartialEq + Eq + Hash>(
    buffer: tt::TokenTreesView<'_, SpanData<Ctx>>,
    span_to_edition: &mut dyn FnMut(Ctx) -> Edition,
) -> parser::Input {
    let mut res = parser::Input::with_capacity(buffer.len());

    let mut current = buffer.cursor();
    let mut syntax_context_to_edition_cache = FxHashMap::default();

    while !current.eof() {
        let tt = current.token_tree();

        // Check if it is lifetime
        if let Some(tt::TokenTree::Leaf(tt::Leaf::Punct(punct))) = tt
            && punct.char == '\''
        {
            current.bump();
            match current.token_tree() {
                Some(tt::TokenTree::Leaf(tt::Leaf::Ident(_ident))) => {
                    res.push(LIFETIME_IDENT);
                    current.bump();
                    continue;
                }
                _ => panic!("Next token must be ident"),
            }
        }

        match tt {
            Some(tt::TokenTree::Leaf(leaf)) => {
                match leaf {
                    tt::Leaf::Literal(lit) => {
                        let kind = match lit.kind {
                            tt::LitKind::Byte => SyntaxKind::BYTE,
                            tt::LitKind::Char => SyntaxKind::CHAR,
                            tt::LitKind::Integer => SyntaxKind::INT_NUMBER,
                            tt::LitKind::Float => SyntaxKind::FLOAT_NUMBER,
                            tt::LitKind::Str | tt::LitKind::StrRaw(_) => SyntaxKind::STRING,
                            tt::LitKind::ByteStr | tt::LitKind::ByteStrRaw(_) => {
                                SyntaxKind::BYTE_STRING
                            }
                            tt::LitKind::CStr | tt::LitKind::CStrRaw(_) => SyntaxKind::C_STRING,
                            tt::LitKind::Err(_) => SyntaxKind::ERROR,
                        };
                        res.push(kind);

                        if kind == FLOAT_NUMBER && !lit.symbol.as_str().ends_with('.') {
                            // Tag the token as joint if it is float with a fractional part
                            // we use this jointness to inform the parser about what token split
                            // event to emit when we encounter a float literal in a field access
                            res.was_joint();
                        }
                    }
                    tt::Leaf::Ident(ident) => {
                        let edition = *syntax_context_to_edition_cache
                            .entry(ident.span.ctx)
                            .or_insert_with(|| span_to_edition(ident.span.ctx));
                        match ident.sym.as_str() {
                            "_" => res.push(T![_]),
                            i if i.starts_with('\'') => res.push(LIFETIME_IDENT),
                            _ if ident.is_raw.yes() => res.push(IDENT),
                            text => match SyntaxKind::from_keyword(text, edition) {
                                Some(kind) => res.push(kind),
                                None => {
                                    let contextual_keyword =
                                        SyntaxKind::from_contextual_keyword(text, edition)
                                            .unwrap_or(SyntaxKind::IDENT);
                                    res.push_ident(contextual_keyword);
                                }
                            },
                        }
                    }
                    tt::Leaf::Punct(punct) => {
                        let kind = SyntaxKind::from_char(punct.char)
                            .unwrap_or_else(|| panic!("{punct:#?} is not a valid punct"));
                        res.push(kind);
                        if punct.spacing == tt::Spacing::Joint {
                            res.was_joint();
                        }
                    }
                }
                current.bump();
            }
            Some(tt::TokenTree::Subtree(subtree)) => {
                if let Some(kind) = match subtree.delimiter.kind {
                    tt::DelimiterKind::Parenthesis => Some(T!['(']),
                    tt::DelimiterKind::Brace => Some(T!['{']),
                    tt::DelimiterKind::Bracket => Some(T!['[']),
                    tt::DelimiterKind::Invisible => None,
                } {
                    res.push(kind);
                }
                current.bump();
            }
            None => {
                let subtree = current.end();
                if let Some(kind) = match subtree.delimiter.kind {
                    tt::DelimiterKind::Parenthesis => Some(T![')']),
                    tt::DelimiterKind::Brace => Some(T!['}']),
                    tt::DelimiterKind::Bracket => Some(T![']']),
                    tt::DelimiterKind::Invisible => None,
                } {
                    res.push(kind);
                }
            }
        };
    }

    res
}
