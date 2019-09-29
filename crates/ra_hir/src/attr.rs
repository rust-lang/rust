use mbe::ast_to_token_tree;
use ra_syntax::{
    ast::{self, AstNode},
    SmolStr,
};
use tt::Subtree;

use crate::{db::AstDatabase, path::Path, Source};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Attr {
    pub(crate) path: Path,
    pub(crate) input: Option<AttrInput>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttrInput {
    Literal(SmolStr),
    TokenTree(Subtree),
}

impl Attr {
    pub(crate) fn from_src(
        Source { file_id, ast }: Source<ast::Attr>,
        db: &impl AstDatabase,
    ) -> Option<Attr> {
        let path = Path::from_src(Source { file_id, ast: ast.path()? }, db)?;
        let input = match ast.input() {
            None => None,
            Some(ast::AttrInput::Literal(lit)) => {
                // FIXME: escape? raw string?
                let value = lit.syntax().first_token()?.text().trim_matches('"').into();
                Some(AttrInput::Literal(value))
            }
            Some(ast::AttrInput::TokenTree(tt)) => {
                Some(AttrInput::TokenTree(ast_to_token_tree(&tt)?.0))
            }
        };

        Some(Attr { path, input })
    }

    pub(crate) fn is_simple_atom(&self, name: &str) -> bool {
        // FIXME: Avoid cloning
        self.path.as_ident().map_or(false, |s| s.to_string() == name)
    }

    pub(crate) fn as_cfg(&self) -> Option<&Subtree> {
        if self.is_simple_atom("cfg") {
            match &self.input {
                Some(AttrInput::TokenTree(subtree)) => Some(subtree),
                _ => None,
            }
        } else {
            None
        }
    }
}
