//! Various traits that are implemented by ast nodes.
//!
//! The implementations are usually trivial, and live in generated.rs

use itertools::Itertools;

use crate::{
    ast::{self, child_opt, children, AstChildren, AstNode, AstToken},
    match_ast,
    syntax_node::{SyntaxElementChildren, SyntaxNodeChildren},
};

pub trait TypeAscriptionOwner: AstNode {
    fn ascribed_type(&self) -> Option<ast::TypeRef> {
        child_opt(self)
    }
}

pub trait NameOwner: AstNode {
    fn name(&self) -> Option<ast::Name> {
        child_opt(self)
    }
}

pub trait VisibilityOwner: AstNode {
    fn visibility(&self) -> Option<ast::Visibility> {
        child_opt(self)
    }
}

pub trait LoopBodyOwner: AstNode {
    fn loop_body(&self) -> Option<ast::BlockExpr> {
        child_opt(self)
    }
}

pub trait ArgListOwner: AstNode {
    fn arg_list(&self) -> Option<ast::ArgList> {
        child_opt(self)
    }
}

pub trait FnDefOwner: AstNode {
    fn functions(&self) -> AstChildren<ast::FnDef> {
        children(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ItemOrMacro {
    Item(ast::ModuleItem),
    Macro(ast::MacroCall),
}

pub trait ModuleItemOwner: AstNode {
    fn items(&self) -> AstChildren<ast::ModuleItem> {
        children(self)
    }
    fn items_with_macros(&self) -> ItemOrMacroIter {
        ItemOrMacroIter(self.syntax().children())
    }
}

#[derive(Debug)]
pub struct ItemOrMacroIter(SyntaxNodeChildren);

impl Iterator for ItemOrMacroIter {
    type Item = ItemOrMacro;
    fn next(&mut self) -> Option<ItemOrMacro> {
        loop {
            let n = self.0.next()?;
            match_ast! {
                match n {
                    ast::ModuleItem(it) => { return Some(ItemOrMacro::Item(it)) },
                    ast::MacroCall(it) => { return Some(ItemOrMacro::Macro(it)) },
                    _ => {},
                }
            }
        }
    }
}

pub trait TypeParamsOwner: AstNode {
    fn type_param_list(&self) -> Option<ast::TypeParamList> {
        child_opt(self)
    }

    fn where_clause(&self) -> Option<ast::WhereClause> {
        child_opt(self)
    }
}

pub trait TypeBoundsOwner: AstNode {
    fn type_bound_list(&self) -> Option<ast::TypeBoundList> {
        child_opt(self)
    }
}

pub trait AttrsOwner: AstNode {
    fn attrs(&self) -> AstChildren<ast::Attr> {
        children(self)
    }
    fn has_atom_attr(&self, atom: &str) -> bool {
        self.attrs().filter_map(|x| x.as_simple_atom()).any(|x| x == atom)
    }
}

pub trait DocCommentsOwner: AstNode {
    fn doc_comments(&self) -> CommentIter {
        CommentIter { iter: self.syntax().children_with_tokens() }
    }

    /// Returns the textual content of a doc comment block as a single string.
    /// That is, strips leading `///` (+ optional 1 character of whitespace),
    /// trailing `*/`, trailing whitespace and then joins the lines.
    fn doc_comment_text(&self) -> Option<String> {
        let mut has_comments = false;
        let docs = self
            .doc_comments()
            .filter(|comment| comment.kind().doc.is_some())
            .map(|comment| {
                has_comments = true;
                let prefix_len = comment.prefix().len();

                let line: &str = comment.text().as_str();

                // Determine if the prefix or prefix + 1 char is stripped
                let pos =
                    if line.chars().nth(prefix_len).map(|c| c.is_whitespace()).unwrap_or(false) {
                        prefix_len + 1
                    } else {
                        prefix_len
                    };

                let end = if comment.kind().shape.is_block() && line.ends_with("*/") {
                    line.len() - 2
                } else {
                    line.len()
                };

                // Note that we do not trim the end of the line here
                // since whitespace can have special meaning at the end
                // of a line in markdown.
                line[pos..end].to_owned()
            })
            .join("\n");

        if has_comments {
            Some(docs)
        } else {
            None
        }
    }
}

pub struct CommentIter {
    iter: SyntaxElementChildren,
}

impl Iterator for CommentIter {
    type Item = ast::Comment;
    fn next(&mut self) -> Option<ast::Comment> {
        self.iter.by_ref().find_map(|el| el.into_token().and_then(ast::Comment::cast))
    }
}
