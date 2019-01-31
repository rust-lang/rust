use crate::{tt, mbe};
use crate::tt_cursor::TtCursor;

/// This module parses a raw `tt::TokenStream` into macro-by-example token
/// stream. This is a *mostly* identify function, expect for handling of
/// `$var:tt_kind` and `$(repeat),*` constructs.

pub fn parse(tt: &tt::Subtree) -> Option<mbe::MacroRules> {
    let mut parser = TtCursor::new(tt);
    let mut rules = Vec::new();
    while !parser.is_eof() {
        rules.push(parse_rule(&mut parser)?)
    }
    Some(mbe::MacroRules { rules })
}

fn parse_rule(p: &mut TtCursor) -> Option<mbe::Rule> {
    let lhs = parse_subtree(p.eat_subtree()?)?;
    p.expect_char('=')?;
    p.expect_char('>')?;
    let rhs = parse_subtree(p.eat_subtree()?)?;
    Some(mbe::Rule { lhs, rhs })
}

fn parse_subtree(tt: &tt::Subtree) -> Option<mbe::Subtree> {
    let mut token_trees = Vec::new();
    let mut p = TtCursor::new(tt);
    while let Some(tt) = p.eat() {
        let child: mbe::TokenTree = match tt {
            tt::TokenTree::Leaf(leaf) => match leaf {
                tt::Leaf::Punct(tt::Punct { char: '$' }) => {
                    if p.at_ident().is_some() {
                        mbe::Leaf::from(parse_var(&mut p)?).into()
                    } else {
                        parse_repeat(&mut p)?.into()
                    }
                }
                tt::Leaf::Punct(tt::Punct { char }) => {
                    mbe::Leaf::from(mbe::Punct { char: *char }).into()
                }
                tt::Leaf::Ident(tt::Ident { text }) => {
                    mbe::Leaf::from(mbe::Ident { text: text.clone() }).into()
                }
                tt::Leaf::Literal(tt::Literal { text }) => {
                    mbe::Leaf::from(mbe::Literal { text: text.clone() }).into()
                }
            },
            tt::TokenTree::Subtree(subtree) => parse_subtree(subtree)?.into(),
        };
        token_trees.push(child);
    }
    Some(mbe::Subtree {
        token_trees,
        delimiter: tt.delimiter,
    })
}

fn parse_var(p: &mut TtCursor) -> Option<mbe::Var> {
    let ident = p.eat_ident().unwrap();
    let text = ident.text.clone();
    let kind = if p.at_char(':') {
        p.bump();
        if let Some(ident) = p.eat_ident() {
            Some(ident.text.clone())
        } else {
            p.rev_bump();
            None
        }
    } else {
        None
    };
    Some(mbe::Var { text, kind })
}

fn parse_repeat(p: &mut TtCursor) -> Option<mbe::Repeat> {
    let subtree = p.eat_subtree().unwrap();
    let subtree = parse_subtree(subtree)?;
    let sep = p.eat_punct()?;
    let (separator, rep) = match sep.char {
        '*' | '+' | '?' => (None, sep.char),
        char => (Some(mbe::Punct { char }), p.eat_punct()?.char),
    };

    let kind = match rep {
        '*' => mbe::RepeatKind::ZeroOrMore,
        '+' => mbe::RepeatKind::OneOrMore,
        '?' => mbe::RepeatKind::ZeroOrOne,
        _ => return None,
    };
    p.bump();
    Some(mbe::Repeat {
        subtree,
        kind,
        separator,
    })
}
