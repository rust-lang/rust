use ra_syntax::ast;

use crate::HirDatabase;

/// Holds documentation
#[derive(Debug, Clone)]
pub struct Documentation(String);

impl Documentation {
    pub fn new(s: &str) -> Self {
        Self(s.into())
    }

    pub fn contents(&self) -> &str {
        &self.0
    }
}

impl Into<String> for Documentation {
    fn into(self) -> String {
        self.contents().into()
    }
}

pub trait Docs {
    fn docs(&self, db: &impl HirDatabase) -> Option<Documentation>;
}

pub(crate) fn docs_from_ast(node: &impl ast::DocCommentsOwner) -> Option<Documentation> {
    let comments = node.doc_comment_text();
    if comments.is_empty() {
        None
    } else {
        Some(Documentation::new(&comments))
    }
}
