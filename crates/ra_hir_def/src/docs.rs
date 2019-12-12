//! Defines hir documentation.
//!
//! This really shouldn't exist, instead, we should deshugar doc comments into attributes, see
//! https://github.com/rust-analyzer/rust-analyzer/issues/2148#issuecomment-550519102

use std::sync::Arc;

use either::Either;
use ra_syntax::ast;

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

    pub fn as_str(&self) -> &str {
        &*self.0
    }

    pub(crate) fn documentation_query(
        db: &impl DefDatabase,
        def: AttrDefId,
    ) -> Option<Documentation> {
        match def {
            AttrDefId::ModuleId(module) => {
                let def_map = db.crate_def_map(module.krate);
                let src = def_map[module.local_id].declaration_source(db)?;
                docs_from_ast(&src.value)
            }
            AttrDefId::StructFieldId(it) => {
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
            AttrDefId::MacroDefId(it) => docs_from_ast(&it.ast_id?.to_node(db)),
            AttrDefId::ConstId(it) => docs_from_ast(&it.lookup(db).source(db).value),
            AttrDefId::StaticId(it) => docs_from_ast(&it.lookup(db).source(db).value),
            AttrDefId::FunctionId(it) => docs_from_ast(&it.lookup(db).source(db).value),
            AttrDefId::TypeAliasId(it) => docs_from_ast(&it.lookup(db).source(db).value),
            AttrDefId::ImplId(_) => None,
        }
    }
}

pub(crate) fn docs_from_ast(node: &impl ast::DocCommentsOwner) -> Option<Documentation> {
    node.doc_comment_text().map(|it| Documentation::new(&it))
}
