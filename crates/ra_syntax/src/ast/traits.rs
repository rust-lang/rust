//! Various traits that are implemented by ast nodes.
//!
//! The implementations are usually trivial, and live in generated.rs
use itertools::Itertools;

use crate::{
    ast::{self, support, AstChildren, AstNode, AstToken},
    syntax_node::SyntaxElementChildren,
    SyntaxToken, T,
};

pub trait NameOwner: AstNode {
    fn name(&self) -> Option<ast::Name> {
        support::child(self.syntax())
    }
}

pub trait VisibilityOwner: AstNode {
    fn visibility(&self) -> Option<ast::Visibility> {
        support::child(self.syntax())
    }
}

pub trait LoopBodyOwner: AstNode {
    fn loop_body(&self) -> Option<ast::BlockExpr> {
        support::child(self.syntax())
    }

    fn label(&self) -> Option<ast::Label> {
        support::child(self.syntax())
    }
}

pub trait ArgListOwner: AstNode {
    fn arg_list(&self) -> Option<ast::ArgList> {
        support::child(self.syntax())
    }
}

pub trait ModuleItemOwner: AstNode {
    fn items(&self) -> AstChildren<ast::Item> {
        support::children(self.syntax())
    }
}

pub trait GenericParamsOwner: AstNode {
    fn generic_param_list(&self) -> Option<ast::GenericParamList> {
        support::child(self.syntax())
    }

    fn where_clause(&self) -> Option<ast::WhereClause> {
        support::child(self.syntax())
    }
}

pub trait TypeBoundsOwner: AstNode {
    fn type_bound_list(&self) -> Option<ast::TypeBoundList> {
        support::child(self.syntax())
    }

    fn colon_token(&self) -> Option<SyntaxToken> {
        support::token(self.syntax(), T![:])
    }
}

pub trait AttrsOwner: AstNode {
    fn attrs(&self) -> AstChildren<ast::Attr> {
        support::children(self.syntax())
    }
    fn has_atom_attr(&self, atom: &str) -> bool {
        self.attrs().filter_map(|x| x.as_simple_atom()).any(|x| x == atom)
    }
}

pub trait DocCommentsOwner: AstNode {
    fn doc_comments(&self) -> CommentIter {
        CommentIter { iter: self.syntax().children_with_tokens() }
    }

    fn doc_comment_text(&self) -> Option<String> {
        self.doc_comments().doc_comment_text()
    }
}

impl CommentIter {
    pub fn from_syntax_node(syntax_node: &ast::SyntaxNode) -> CommentIter {
        CommentIter { iter: syntax_node.children_with_tokens() }
    }

    /// Returns the textual content of a doc comment block as a single string.
    /// That is, strips leading `///` (+ optional 1 character of whitespace),
    /// trailing `*/`, trailing whitespace and then joins the lines.
    pub fn doc_comment_text(self) -> Option<String> {
        let mut has_comments = false;
        let docs = self
            .filter(|comment| comment.kind().doc.is_some())
            .map(|comment| {
                has_comments = true;
                let prefix_len = comment.prefix().len();

                let line: &str = comment.text().as_str();

                // Determine if the prefix or prefix + 1 char is stripped
                let pos =
                    if let Some(ws) = line.chars().nth(prefix_len).filter(|c| c.is_whitespace()) {
                        prefix_len + ws.len_utf8()
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
