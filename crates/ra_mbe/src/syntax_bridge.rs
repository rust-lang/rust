use ra_syntax::{ast, AstNode, SyntaxNode, SyntaxKind::*};

pub fn ast_to_token_tree(ast: &ast::TokenTree) -> Option<tt::Subtree> {
    convert_tt(ast.syntax())
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
                        tt::Leaf::from(tt::Punct { char, spacing: tt::Spacing::Joint }).into(),
                    );
                }
                prev = Some(char)
            }
            if let Some(char) = prev {
                token_trees
                    .push(tt::Leaf::from(tt::Punct { char, spacing: tt::Spacing::Alone }).into());
            }
        } else {
            let child: tt::TokenTree = if child.kind() == TOKEN_TREE {
                convert_tt(child)?.into()
            } else if child.kind().is_keyword() || child.kind() == IDENT {
                let text = child.leaf_text().unwrap().clone();
                tt::Leaf::from(tt::Ident { text }).into()
            } else if child.kind().is_literal() {
                tt::Leaf::from(tt::Literal { text: child.leaf_text().unwrap().clone() }).into()
            } else {
                return None;
            };
            token_trees.push(child)
        }
    }

    let res = tt::Subtree { delimiter, token_trees };
    Some(res)
}
