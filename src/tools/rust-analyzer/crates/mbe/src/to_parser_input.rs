//! Convert macro-by-example tokens which are specific to macro expansion into a
//! format that works for our parser.

use syntax::{SyntaxKind, SyntaxKind::*, T};

use crate::tt::buffer::TokenBuffer;

pub(crate) fn to_parser_input(buffer: &TokenBuffer<'_>) -> parser::Input {
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
                        let is_negated = lit.text.starts_with('-');
                        let inner_text = &lit.text[if is_negated { 1 } else { 0 }..];

                        let kind = parser::LexedStr::single_token(inner_text)
                            .map(|(kind, _error)| kind)
                            .filter(|kind| {
                                kind.is_literal()
                                    && (!is_negated || matches!(kind, FLOAT_NUMBER | INT_NUMBER))
                            })
                            .unwrap_or_else(|| panic!("Fail to convert given literal {:#?}", &lit));

                        res.push(kind);

                        if kind == FLOAT_NUMBER && !inner_text.ends_with('.') {
                            // Tag the token as joint if it is float with a fractional part
                            // we use this jointness to inform the parser about what token split
                            // event to emit when we encounter a float literal in a field access
                            res.was_joint();
                        }
                    }
                    tt::Leaf::Ident(ident) => match ident.text.as_ref() {
                        "_" => res.push(T![_]),
                        i if i.starts_with('\'') => res.push(LIFETIME_IDENT),
                        _ => match SyntaxKind::from_keyword(&ident.text) {
                            Some(kind) => res.push(kind),
                            None => {
                                let contextual_keyword =
                                    SyntaxKind::from_contextual_keyword(&ident.text)
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
