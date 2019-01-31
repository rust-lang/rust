use crate::tt_cursor::TtCursor;

/// This module parses a raw `tt::TokenStream` into macro-by-example token
/// stream. This is a *mostly* identify function, expect for handling of
/// `$var:tt_kind` and `$(repeat),*` constructs.

pub fn parse(tt: &tt::Subtree) -> Option<crate::MacroRules> {
    let mut parser = TtCursor::new(tt);
    let mut rules = Vec::new();
    while !parser.is_eof() {
        rules.push(parse_rule(&mut parser)?)
    }
    Some(crate::MacroRules { rules })
}

fn parse_rule(p: &mut TtCursor) -> Option<crate::Rule> {
    let lhs = parse_subtree(p.eat_subtree()?)?;
    p.expect_char('=')?;
    p.expect_char('>')?;
    let mut rhs = parse_subtree(p.eat_subtree()?)?;
    rhs.delimiter = crate::Delimiter::None;
    Some(crate::Rule { lhs, rhs })
}

fn parse_subtree(tt: &tt::Subtree) -> Option<crate::Subtree> {
    let mut token_trees = Vec::new();
    let mut p = TtCursor::new(tt);
    while let Some(tt) = p.eat() {
        let child: crate::TokenTree = match tt {
            tt::TokenTree::Leaf(leaf) => match leaf {
                tt::Leaf::Punct(tt::Punct { char: '$', .. }) => {
                    if p.at_ident().is_some() {
                        crate::Leaf::from(parse_var(&mut p)?).into()
                    } else {
                        parse_repeat(&mut p)?.into()
                    }
                }
                tt::Leaf::Punct(punct) => crate::Leaf::from(*punct).into(),
                tt::Leaf::Ident(tt::Ident { text }) => {
                    crate::Leaf::from(crate::Ident { text: text.clone() }).into()
                }
                tt::Leaf::Literal(tt::Literal { text }) => {
                    crate::Leaf::from(crate::Literal { text: text.clone() }).into()
                }
            },
            tt::TokenTree::Subtree(subtree) => parse_subtree(&subtree)?.into(),
        };
        token_trees.push(child);
    }
    Some(crate::Subtree {
        token_trees,
        delimiter: tt.delimiter,
    })
}

fn parse_var(p: &mut TtCursor) -> Option<crate::Var> {
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
    Some(crate::Var { text, kind })
}

fn parse_repeat(p: &mut TtCursor) -> Option<crate::Repeat> {
    let subtree = p.eat_subtree().unwrap();
    let mut subtree = parse_subtree(subtree)?;
    subtree.delimiter = crate::Delimiter::None;
    let sep = p.eat_punct()?;
    let (separator, rep) = match sep.char {
        '*' | '+' | '?' => (None, sep.char),
        char => (Some(char), p.eat_punct()?.char),
    };

    let kind = match rep {
        '*' => crate::RepeatKind::ZeroOrMore,
        '+' => crate::RepeatKind::OneOrMore,
        '?' => crate::RepeatKind::ZeroOrOne,
        _ => return None,
    };
    p.bump();
    Some(crate::Repeat {
        subtree,
        kind,
        separator,
    })
}
