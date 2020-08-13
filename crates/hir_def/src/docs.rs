//! Defines hir documentation.
//!
//! This really shouldn't exist, instead, we should deshugar doc comments into attributes, see
//! https://github.com/rust-analyzer/rust-analyzer/issues/2148#issuecomment-550519102

use std::sync::Arc;

use either::Either;
use syntax::ast;

use crate::{
    db::DefDatabase,
    src::{HasChildSource, HasSource},
    AdtId, AttrDefId, Lookup,
};

/// Holds documentation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Documentation(Arc<str>);

impl Into<String> for Documentation {
    fn into(self) -> String {
        self.as_str().to_owned()
    }
}

impl Documentation {
    fn new(s: &str) -> Documentation {
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

    pub(crate) fn documentation_query(
        db: &dyn DefDatabase,
        def: AttrDefId,
    ) -> Option<Documentation> {
        match def {
            AttrDefId::ModuleId(module) => {
                let def_map = db.crate_def_map(module.krate);
                let src = def_map[module.local_id].declaration_source(db)?;
                docs_from_ast(&src.value)
            }
            AttrDefId::FieldId(it) => {
                let src = it.parent.child_source(db);
                match &src.value[it.local_id] {
                    Either::Left(_tuple) => None,
                    Either::Right(record) => docs_from_ast(record),
                }
            }
            AttrDefId::AdtId(it) => match it {
                AdtId::StructId(it) => docs_from_ast(&it.lookup(db).source(db).value),
                AdtId::EnumId(it) => docs_from_ast(&it.lookup(db).source(db).value),
                AdtId::UnionId(it) => docs_from_ast(&it.lookup(db).source(db).value),
            },
            AttrDefId::EnumVariantId(it) => {
                let src = it.parent.child_source(db);
                docs_from_ast(&src.value[it.local_id])
            }
            AttrDefId::TraitId(it) => docs_from_ast(&it.lookup(db).source(db).value),
            AttrDefId::MacroDefId(it) => docs_from_ast(&it.ast_id?.to_node(db.upcast())),
            AttrDefId::ConstId(it) => docs_from_ast(&it.lookup(db).source(db).value),
            AttrDefId::StaticId(it) => docs_from_ast(&it.lookup(db).source(db).value),
            AttrDefId::FunctionId(it) => docs_from_ast(&it.lookup(db).source(db).value),
            AttrDefId::TypeAliasId(it) => docs_from_ast(&it.lookup(db).source(db).value),
            AttrDefId::ImplId(_) => None,
        }
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
            comment_text.push_str("\n\n");
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
    for attr in owner.attrs() {
        if let Some(("doc", value)) =
            attr.as_simple_key_value().as_ref().map(|(k, v)| (k.as_str(), v.as_str()))
        {
            docs.push_str(value);
            docs.push_str("\n\n");
        }
    }
    if docs.is_empty() {
        None
    } else {
        Some(docs.trim_end_matches("\n\n").to_owned())
    }
}
