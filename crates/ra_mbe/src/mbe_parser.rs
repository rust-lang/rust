/// This module parses a raw `tt::TokenStream` into macro-by-example token
/// stream. This is a *mostly* identify function, expect for handling of
/// `$var:tt_kind` and `$(repeat),*` constructs.
use crate::ParseError;
use crate::tt_cursor::TtCursor;

pub(crate) fn parse(tt: &tt::Subtree) -> Result<crate::MacroRules, ParseError> {
    let mut parser = TtCursor::new(tt);
    let mut rules = Vec::new();
    while !parser.is_eof() {
        rules.push(parse_rule(&mut parser)?);
        if let Err(e) = parser.expect_char(';') {
            if !parser.is_eof() {
                return Err(e);
            }
            break;
        }
    }
    Ok(crate::MacroRules { rules })
}

fn parse_rule(p: &mut TtCursor) -> Result<crate::Rule, ParseError> {
    let lhs = parse_subtree(p.eat_subtree()?)?;
    p.expect_char('=')?;
    p.expect_char('>')?;
    let mut rhs = parse_subtree(p.eat_subtree()?)?;
    rhs.delimiter = crate::Delimiter::None;
    Ok(crate::Rule { lhs, rhs })
}

fn parse_subtree(tt: &tt::Subtree) -> Result<crate::Subtree, ParseError> {
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
                tt::Leaf::Ident(tt::Ident { text, id: _ }) => {
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
    Ok(crate::Subtree { token_trees, delimiter: tt.delimiter })
}

fn parse_var(p: &mut TtCursor) -> Result<crate::Var, ParseError> {
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
    Ok(crate::Var { text, kind })
}

fn parse_repeat(p: &mut TtCursor) -> Result<crate::Repeat, ParseError> {
    let subtree = p.eat_subtree().unwrap();
    let mut subtree = parse_subtree(subtree)?;
    subtree.delimiter = crate::Delimiter::None;
    let sep = p.eat_punct().ok_or(ParseError::Expected(String::from("separator")))?;
    let (separator, rep) = match sep.char {
        '*' | '+' | '?' => (None, sep.char),
        char => {
            (Some(char), p.eat_punct().ok_or(ParseError::Expected(String::from("separator")))?.char)
        }
    };

    let kind = match rep {
        '*' => crate::RepeatKind::ZeroOrMore,
        '+' => crate::RepeatKind::OneOrMore,
        '?' => crate::RepeatKind::ZeroOrOne,
        _ => return Err(ParseError::Expected(String::from("repeat"))),
    };
    p.bump();
    Ok(crate::Repeat { subtree, kind, separator })
}

#[cfg(test)]
mod tests {
    use ra_syntax::{ast, AstNode};

    use super::*;
    use crate::ast_to_token_tree;

    #[test]
    fn test_invalid_parse() {
        expect_err("invalid", "subtree");

        is_valid("($i:ident) => ()");
        expect_err("$i:ident => ()", "subtree");
        expect_err("($i:ident) ()", "`=`");
        expect_err("($($i:ident)_) => ()", "separator");
    }

    fn expect_err(macro_body: &str, expected: &str) {
        assert_eq!(
            create_rules(&format_macro(macro_body)),
            Err(ParseError::Expected(String::from(expected)))
        );
    }

    fn is_valid(macro_body: &str) {
        assert!(create_rules(&format_macro(macro_body)).is_ok());
    }

    fn format_macro(macro_body: &str) -> String {
        format!(
            "
        macro_rules! foo {{
            {}
        }}
",
            macro_body
        )
    }

    fn create_rules(macro_definition: &str) -> Result<crate::MacroRules, ParseError> {
        let source_file = ast::SourceFile::parse(macro_definition);
        let macro_definition =
            source_file.syntax().descendants().find_map(ast::MacroCall::cast).unwrap();

        let (definition_tt, _) = ast_to_token_tree(macro_definition.token_tree().unwrap()).unwrap();
        parse(&definition_tt)
    }

}
