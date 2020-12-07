//! Defines hir documentation.
//!
//! This really shouldn't exist, instead, we should deshugar doc comments into attributes, see
//! https://github.com/rust-analyzer/rust-analyzer/issues/2148#issuecomment-550519102

use std::sync::Arc;

use itertools::Itertools;
use syntax::{ast, SmolStr};

/// Holds documentation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Documentation(Arc<str>);

impl Into<String> for Documentation {
    fn into(self) -> String {
        self.as_str().to_owned()
    }
}

impl Documentation {
    pub fn new(s: &str) -> Documentation {
        Documentation(s.into())
    }

    pub fn from_ast<N>(node: &N) -> Option<Documentation>
    where
        N: ast::DocCommentsOwner + ast::AttrsOwner,
    {
        docs_from_ast(node)
    }

    pub fn as_str(&self) -> &str {
        &*self.0
    }
}

pub(crate) fn docs_from_ast<N>(node: &N) -> Option<Documentation>
where
    N: ast::DocCommentsOwner + ast::AttrsOwner,
{
    let doc_comment_text = node.doc_comment_text();
    let doc_attr_text = expand_doc_attrs(node);
    let docs = merge_doc_comments_and_attrs(doc_comment_text, doc_attr_text);
    docs.map(|it| Documentation::new(&it))
}

fn merge_doc_comments_and_attrs(
    doc_comment_text: Option<String>,
    doc_attr_text: Option<String>,
) -> Option<String> {
    match (doc_comment_text, doc_attr_text) {
        (Some(mut comment_text), Some(attr_text)) => {
            comment_text.reserve(attr_text.len() + 1);
            comment_text.push('\n');
            comment_text.push_str(&attr_text);
            Some(comment_text)
        }
        (Some(comment_text), None) => Some(comment_text),
        (None, Some(attr_text)) => Some(attr_text),
        (None, None) => None,
    }
}

fn expand_doc_attrs(owner: &dyn ast::AttrsOwner) -> Option<String> {
    let mut docs = String::new();
    owner
        .attrs()
        .filter_map(|attr| attr.as_simple_key_value().filter(|(key, _)| key == "doc"))
        .map(|(_, value)| value)
        .intersperse(SmolStr::new_inline("\n"))
        // No FromIterator<SmolStr> for String
        .for_each(|s| docs.push_str(s.as_str()));
    if docs.is_empty() {
        None
    } else {
        Some(docs)
    }
}
