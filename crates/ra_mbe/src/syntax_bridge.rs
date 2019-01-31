use ra_syntax::{ast, AstNode, SyntaxNode, SyntaxKind::*};

pub fn macro_call_to_tt(call: &ast::MacroCall) -> Option<tt::Subtree> {
    let tt = call.token_tree()?;
    convert_tt(tt.syntax())
}

fn convert_tt(tt: &SyntaxNode) -> Option<tt::Subtree> {
    let first_child = tt.first_child()?;
    let last_child = tt.last_child()?;
    let delimiter = match (first_child.kind(), last_child.kind()) {
        (L_PAREN, R_PAREN) => tt::Delimiter::Parenthesis,
        (L_CURLY, R_CURLY) => tt::Delimiter::Brace,
        (L_BRACK, R_BRACK) => tt::Delimiter::Bracket,
        _ => return None,
    };
    let mut token_trees = Vec::new();
    for child in tt.children().skip(1) {
        if child == first_child || child == last_child || child.kind().is_trivia() {
            continue;
        }
        if child.kind().is_punct() {
            let mut prev = None;
            for char in child.leaf_text().unwrap().chars() {
                if let Some(char) = prev {
                    token_trees.push(
                        tt::Leaf::from(tt::Punct {
                            char,
                            spacing: tt::Spacing::Joint,
                        })
                        .into(),
                    );
                }
                prev = Some(char)
            }
            if let Some(char) = prev {
                token_trees.push(
                    tt::Leaf::from(tt::Punct {
                        char,
                        spacing: tt::Spacing::Alone,
                    })
                    .into(),
                );
            }
        } else {
            let child: tt::TokenTree = if child.kind() == TOKEN_TREE {
                convert_tt(child)?.into()
            } else if child.kind().is_keyword() || child.kind() == IDENT {
                let text = child.leaf_text().unwrap().clone();
                tt::Leaf::from(tt::Ident { text }).into()
            } else if child.kind().is_literal() {
                tt::Leaf::from(tt::Literal {
                    text: child.leaf_text().unwrap().clone(),
                })
                .into()
            } else {
                return None;
            };
            token_trees.push(child)
        }
    }

    let res = tt::Subtree {
        delimiter,
        token_trees,
    };
    Some(res)
}

#[test]
fn test_convert_tt() {
    let macro_definition = r#"
macro_rules! impl_froms {
    ($e:ident: $($v:ident),*) => {
        $(
            impl From<$v> for $e {
                fn from(it: $v) -> $e {
                    $e::$v(it)
                }
            }
        )*
    }
}
"#;

    let macro_invocation = r#"
impl_froms!(TokenTree: Leaf, Subtree);
"#;

    let source_file = ast::SourceFile::parse(macro_definition);
    let macro_definition = source_file
        .syntax()
        .descendants()
        .find_map(ast::MacroCall::cast)
        .unwrap();

    let source_file = ast::SourceFile::parse(macro_invocation);
    let macro_invocation = source_file
        .syntax()
        .descendants()
        .find_map(ast::MacroCall::cast)
        .unwrap();

    let definition_tt = macro_call_to_tt(macro_definition).unwrap();
    let invocation_tt = macro_call_to_tt(macro_invocation).unwrap();
    let mbe = crate::parse(&definition_tt).unwrap();
    let expansion = crate::exapnd(&mbe, &invocation_tt).unwrap();
    assert_eq!(
        expansion.to_string(),
        "{(impl From < Leaf > for TokenTree {fn from (it : Leaf) -> TokenTree {TokenTree :: Leaf (it)}}) \
          (impl From < Subtree > for TokenTree {fn from (it : Subtree) -> TokenTree {TokenTree :: Subtree (it)}})}"
    )
}
