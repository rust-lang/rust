use syntax::{ast, AstNode};

use crate::ast_to_token_tree;

use super::*;

#[test]
fn test_valid_arms() {
    fn check(macro_body: &str) {
        let m = parse_macro_arm(macro_body);
        m.unwrap();
    }

    check("($i:ident) => ()");
    check("($(x),*) => ()");
    check("($(x)_*) => ()");
    check("($(x)i*) => ()");
    check("($($i:ident)*) => ($_)");
    check("($($true:ident)*) => ($true)");
    check("($($false:ident)*) => ($false)");
    check("($) => ($)");
}

#[test]
fn test_invalid_arms() {
    fn check(macro_body: &str, err: ParseError) {
        let m = parse_macro_arm(macro_body);
        assert_eq!(m, Err(err));
    }
    check("invalid", ParseError::Expected("expected subtree".into()));

    check("$i:ident => ()", ParseError::Expected("expected subtree".into()));
    check("($i:ident) ()", ParseError::Expected("expected `=`".into()));
    check("($($i:ident)_) => ()", ParseError::InvalidRepeat);

    check("($i) => ($i)", ParseError::UnexpectedToken("bad fragment specifier 1".into()));
    check("($i:) => ($i)", ParseError::UnexpectedToken("bad fragment specifier 1".into()));
    check("($i:_) => ()", ParseError::UnexpectedToken("bad fragment specifier 1".into()));
}

fn parse_macro_arm(arm_definition: &str) -> Result<crate::MacroRules, ParseError> {
    let macro_definition = format!(" macro_rules! m {{ {} }} ", arm_definition);
    let source_file = ast::SourceFile::parse(&macro_definition).ok().unwrap();
    let macro_definition =
        source_file.syntax().descendants().find_map(ast::MacroRules::cast).unwrap();

    let (definition_tt, _) = ast_to_token_tree(&macro_definition.token_tree().unwrap()).unwrap();
    crate::MacroRules::parse(&definition_tt)
}
