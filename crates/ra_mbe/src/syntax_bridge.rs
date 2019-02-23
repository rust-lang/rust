use ra_parser::TokenSource;
use ra_syntax::{
    AstNode, SyntaxNode, TextRange, SyntaxKind,
    ast, SyntaxKind::*, TextUnit
};

/// Maps `tt::TokenId` to the relative range of the original token.
#[derive(Default)]
pub struct TokenMap {
    /// Maps `tt::TokenId` to the *relative* source range.
    toknes: Vec<TextRange>,
}

/// Convert the syntax tree (what user has written) to a `TokenTree` (what macro
/// will consume).
pub fn ast_to_token_tree(ast: &ast::TokenTree) -> Option<(tt::Subtree, TokenMap)> {
    let mut token_map = TokenMap::default();
    let node = ast.syntax();
    let tt = convert_tt(&mut token_map, node.range().start(), node)?;
    Some((tt, token_map))
}

/// Parses the token tree (result of macro expansion) as a sequence of items
pub fn token_tree_to_ast_item_list(tt: &tt::Subtree) -> ast::SourceFile {
    unimplemented!()
}

impl TokenMap {
    pub fn relative_range_of(&self, tt: tt::TokenId) -> Option<TextRange> {
        let idx = tt.0 as usize;
        self.toknes.get(idx).map(|&it| it)
    }

    fn alloc(&mut self, relative_range: TextRange) -> tt::TokenId {
        let id = self.toknes.len();
        self.toknes.push(relative_range);
        tt::TokenId(id as u32)
    }
}

fn convert_tt(
    token_map: &mut TokenMap,
    global_offset: TextUnit,
    tt: &SyntaxNode,
) -> Option<tt::Subtree> {
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
                convert_tt(token_map, global_offset, child)?.into()
            } else if child.kind().is_keyword() || child.kind() == IDENT {
                let relative_range = child.range() - global_offset;
                let id = token_map.alloc(relative_range);
                let text = child.leaf_text().unwrap().clone();
                tt::Leaf::from(tt::Ident { text, id }).into()
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

struct TtTokenSource;

impl TtTokenSource {
    fn new(tt: &tt::Subtree) -> TtTokenSource {
        unimplemented!()
    }
}

impl TokenSource for TtTokenSource {
    fn token_kind(&self, pos: usize) -> SyntaxKind {
        unimplemented!()
    }
    fn is_token_joint_to_next(&self, pos: usize) -> bool {
        unimplemented!()
    }
    fn is_keyword(&self, pos: usize, kw: &str) -> bool {
        unimplemented!()
    }
}
