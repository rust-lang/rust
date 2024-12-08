//! Convert macro-by-example tokens which are specific to macro expansion into a
//! format that works for our parser.

use std::fmt;

use span::Edition;
use syntax::{SyntaxKind, SyntaxKind::*, T};

use tt::buffer::TokenBuffer;

pub fn to_parser_input<S: Copy + fmt::Debug>(
    edition: Edition,
    buffer: &TokenBuffer<'_, S>,
) -> parser::Input {
    let mut res = parser::Input::default();

    let mut current = buffer.begin();

    while !current.eof() {
        let cursor = current;
        let tt = cursor.token_tree();

        // Check if it is lifetime
        if let Some(tt::buffer::TokenTreeRef::Leaf(tt::Leaf::Punct(punct), _)) = tt {
            if punct.char == '\'' {
                let next = cursor.bump();
                match next.token_tree() {
                    Some(tt::buffer::TokenTreeRef::Leaf(tt::Leaf::Ident(_ident), _)) => {
                        res.push(LIFETIME_IDENT);
                        current = next.bump();
                        continue;
                    }
                    _ => panic!("Next token must be ident : {:#?}", next.token_tree()),
                }
            }
        }

        current = match tt {
            Some(tt::buffer::TokenTreeRef::Leaf(leaf, _)) => {
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
                    tt::Leaf::Ident(ident) => match ident.sym.as_str() {
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
                    },
                    tt::Leaf::Punct(punct) => {
                        let kind = SyntaxKind::from_char(punct.char)
                            .unwrap_or_else(|| panic!("{punct:#?} is not a valid punct"));
                        res.push(kind);
                        if punct.spacing == tt::Spacing::Joint {
                            res.was_joint();
                        }
                    }
                }
                cursor.bump()
            }
            Some(tt::buffer::TokenTreeRef::Subtree(subtree, _)) => {
                if let Some(kind) = match subtree.delimiter.kind {
                    tt::DelimiterKind::Parenthesis => Some(T!['(']),
                    tt::DelimiterKind::Brace => Some(T!['{']),
                    tt::DelimiterKind::Bracket => Some(T!['[']),
                    tt::DelimiterKind::Invisible => None,
                } {
                    res.push(kind);
                }
                cursor.subtree().unwrap()
            }
            None => match cursor.end() {
                Some(subtree) => {
                    if let Some(kind) = match subtree.delimiter.kind {
                        tt::DelimiterKind::Parenthesis => Some(T![')']),
                        tt::DelimiterKind::Brace => Some(T!['}']),
                        tt::DelimiterKind::Bracket => Some(T![']']),
                        tt::DelimiterKind::Invisible => None,
                    } {
                        res.push(kind);
                    }
                    cursor.bump()
                }
                None => continue,
            },
        };
    }

    res
}
