mod expand;
mod rule;

use std::fmt::Write;

use ::parser::FragmentKind;
use syntax::{ast, AstNode, NodeOrToken, SyntaxNode, WalkEvent};
use test_utils::assert_eq_text;

use super::*;

pub(crate) struct MacroFixture {
    rules: MacroRules,
}

pub(crate) struct MacroFixture2 {
    rules: MacroDef,
}

macro_rules! impl_fixture {
    ($name:ident) => {
        impl $name {
            pub(crate) fn expand_tt(&self, invocation: &str) -> tt::Subtree {
                self.try_expand_tt(invocation).unwrap()
            }

            fn try_expand_tt(&self, invocation: &str) -> Result<tt::Subtree, ExpandError> {
                let source_file = ast::SourceFile::parse(invocation).tree();
                let macro_invocation =
                    source_file.syntax().descendants().find_map(ast::MacroCall::cast).unwrap();

                let (invocation_tt, _) = ast_to_token_tree(&macro_invocation.token_tree().unwrap())
                    .ok_or_else(|| ExpandError::ConversionError)?;

                self.rules.expand(&invocation_tt).result()
            }

            #[allow(unused)]
            fn assert_expand_err(&self, invocation: &str, err: &ExpandError) {
                assert_eq!(self.try_expand_tt(invocation).as_ref(), Err(err));
            }

            #[allow(unused)]
            fn expand_items(&self, invocation: &str) -> SyntaxNode {
                let expanded = self.expand_tt(invocation);
                token_tree_to_syntax_node(&expanded, FragmentKind::Items).unwrap().0.syntax_node()
            }

            #[allow(unused)]
            fn expand_statements(&self, invocation: &str) -> SyntaxNode {
                let expanded = self.expand_tt(invocation);
                token_tree_to_syntax_node(&expanded, FragmentKind::Statements)
                    .unwrap()
                    .0
                    .syntax_node()
            }

            #[allow(unused)]
            fn expand_expr(&self, invocation: &str) -> SyntaxNode {
                let expanded = self.expand_tt(invocation);
                token_tree_to_syntax_node(&expanded, FragmentKind::Expr).unwrap().0.syntax_node()
            }

            #[allow(unused)]
            fn assert_expand_tt(&self, invocation: &str, expected: &str) {
                let expansion = self.expand_tt(invocation);
                assert_eq!(expansion.to_string(), expected);
            }

            #[allow(unused)]
            fn assert_expand(&self, invocation: &str, expected: &str) {
                let expansion = self.expand_tt(invocation);
                let actual = format!("{:?}", expansion);
                test_utils::assert_eq_text!(&expected.trim(), &actual.trim());
            }

            fn assert_expand_items(&self, invocation: &str, expected: &str) -> &$name {
                self.assert_expansion(FragmentKind::Items, invocation, expected);
                self
            }

            #[allow(unused)]
            fn assert_expand_statements(&self, invocation: &str, expected: &str) -> &$name {
                self.assert_expansion(FragmentKind::Statements, invocation, expected);
                self
            }

            fn assert_expansion(&self, kind: FragmentKind, invocation: &str, expected: &str) {
                let expanded = self.expand_tt(invocation);
                assert_eq!(expanded.to_string(), expected);

                let expected = expected.replace("$crate", "C_C__C");

                // wrap the given text to a macro call
                let expected = {
                    let wrapped = format!("wrap_macro!( {} )", expected);
                    let wrapped = ast::SourceFile::parse(&wrapped);
                    let wrapped = wrapped
                        .tree()
                        .syntax()
                        .descendants()
                        .find_map(ast::TokenTree::cast)
                        .unwrap();
                    let mut wrapped = ast_to_token_tree(&wrapped).unwrap().0;
                    wrapped.delimiter = None;
                    wrapped
                };

                let expanded_tree =
                    token_tree_to_syntax_node(&expanded, kind).unwrap().0.syntax_node();
                let expanded_tree = debug_dump_ignore_spaces(&expanded_tree).trim().to_string();

                let expected_tree =
                    token_tree_to_syntax_node(&expected, kind).unwrap().0.syntax_node();
                let expected_tree = debug_dump_ignore_spaces(&expected_tree).trim().to_string();

                let expected_tree = expected_tree.replace("C_C__C", "$crate");
                assert_eq!(
                    expanded_tree, expected_tree,
                    "\nleft:\n{}\nright:\n{}",
                    expanded_tree, expected_tree,
                );
            }
        }
    };
}

impl_fixture!(MacroFixture);
impl_fixture!(MacroFixture2);

pub(crate) fn parse_macro(ra_fixture: &str) -> MacroFixture {
    let definition_tt = parse_macro_rules_to_tt(ra_fixture);
    let rules = MacroRules::parse(&definition_tt).unwrap();
    MacroFixture { rules }
}

pub(crate) fn parse_macro2(ra_fixture: &str) -> MacroFixture2 {
    let definition_tt = parse_macro_def_to_tt(ra_fixture);
    let rules = MacroDef::parse(&definition_tt).unwrap();
    MacroFixture2 { rules }
}

pub(crate) fn parse_macro_error(ra_fixture: &str) -> ParseError {
    let definition_tt = parse_macro_rules_to_tt(ra_fixture);

    match MacroRules::parse(&definition_tt) {
        Ok(_) => panic!("Expect error"),
        Err(err) => err,
    }
}

pub(crate) fn parse_to_token_tree_by_syntax(ra_fixture: &str) -> tt::Subtree {
    let source_file = ast::SourceFile::parse(ra_fixture).ok().unwrap();
    let tt = syntax_node_to_token_tree(source_file.syntax()).unwrap().0;

    let parsed = parse_to_token_tree(ra_fixture).unwrap().0;
    assert_eq!(tt, parsed);

    parsed
}

fn parse_macro_rules_to_tt(ra_fixture: &str) -> tt::Subtree {
    let source_file = ast::SourceFile::parse(ra_fixture).ok().unwrap();
    let macro_definition =
        source_file.syntax().descendants().find_map(ast::MacroRules::cast).unwrap();

    let (definition_tt, _) = ast_to_token_tree(&macro_definition.token_tree().unwrap()).unwrap();

    let parsed = parse_to_token_tree(
        &ra_fixture[macro_definition.token_tree().unwrap().syntax().text_range()],
    )
    .unwrap()
    .0;
    assert_eq!(definition_tt, parsed);

    definition_tt
}

fn parse_macro_def_to_tt(ra_fixture: &str) -> tt::Subtree {
    let source_file = ast::SourceFile::parse(ra_fixture).ok().unwrap();
    let macro_definition =
        source_file.syntax().descendants().find_map(ast::MacroDef::cast).unwrap();

    let (definition_tt, _) = ast_to_token_tree(&macro_definition.body().unwrap()).unwrap();

    let parsed =
        parse_to_token_tree(&ra_fixture[macro_definition.body().unwrap().syntax().text_range()])
            .unwrap()
            .0;
    assert_eq!(definition_tt, parsed);

    definition_tt
}

fn debug_dump_ignore_spaces(node: &syntax::SyntaxNode) -> String {
    let mut level = 0;
    let mut buf = String::new();
    macro_rules! indent {
        () => {
            for _ in 0..level {
                buf.push_str("  ");
            }
        };
    }

    for event in node.preorder_with_tokens() {
        match event {
            WalkEvent::Enter(element) => {
                match element {
                    NodeOrToken::Node(node) => {
                        indent!();
                        writeln!(buf, "{:?}", node.kind()).unwrap();
                    }
                    NodeOrToken::Token(token) => match token.kind() {
                        syntax::SyntaxKind::WHITESPACE => {}
                        _ => {
                            indent!();
                            writeln!(buf, "{:?}", token.kind()).unwrap();
                        }
                    },
                }
                level += 1;
            }
            WalkEvent::Leave(_) => level -= 1,
        }
    }

    buf
}
