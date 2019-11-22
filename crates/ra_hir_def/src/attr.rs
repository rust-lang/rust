//! A higher level attributes based on TokenTree, with also some shortcuts.

use std::{ops, sync::Arc};

use hir_expand::hygiene::Hygiene;
use mbe::ast_to_token_tree;
use ra_cfg::CfgOptions;
use ra_syntax::{
    ast::{self, AstNode, AttrsOwner},
    SmolStr,
};
use tt::Subtree;

use crate::path::Path;

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct Attrs {
    entries: Option<Arc<[Attr]>>,
}

impl ops::Deref for Attrs {
    type Target = [Attr];

    fn deref(&self) -> &[Attr] {
        match &self.entries {
            Some(it) => &*it,
            None => &[],
        }
    }
}

impl Attrs {
    pub fn has_atom(&self, atom: &str) -> bool {
        self.iter().any(|it| it.is_simple_atom(atom))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attr {
    pub(crate) path: Path,
    pub(crate) input: Option<AttrInput>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttrInput {
    Literal(SmolStr),
    TokenTree(Subtree),
}

impl Attr {
    pub(crate) fn from_src(ast: ast::Attr, hygiene: &Hygiene) -> Option<Attr> {
        let path = Path::from_src(ast.path()?, hygiene)?;
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

    pub fn from_attrs_owner(owner: &dyn AttrsOwner, hygiene: &Hygiene) -> Attrs {
        let mut attrs = owner.attrs().peekable();
        let entries = if attrs.peek().is_none() {
            // Avoid heap allocation
            None
        } else {
            Some(attrs.flat_map(|ast| Attr::from_src(ast, hygiene)).collect())
        };
        Attrs { entries }
    }

    pub fn is_simple_atom(&self, name: &str) -> bool {
        // FIXME: Avoid cloning
        self.path.as_ident().map_or(false, |s| s.to_string() == name)
    }

    // FIXME: handle cfg_attr :-)
    pub fn as_cfg(&self) -> Option<&Subtree> {
        if !self.is_simple_atom("cfg") {
            return None;
        }
        match &self.input {
            Some(AttrInput::TokenTree(subtree)) => Some(subtree),
            _ => None,
        }
    }

    pub fn as_path(&self) -> Option<&SmolStr> {
        if !self.is_simple_atom("path") {
            return None;
        }
        match &self.input {
            Some(AttrInput::Literal(it)) => Some(it),
            _ => None,
        }
    }

    pub fn is_cfg_enabled(&self, cfg_options: &CfgOptions) -> Option<bool> {
        cfg_options.is_cfg_enabled(self.as_cfg()?)
    }
}
