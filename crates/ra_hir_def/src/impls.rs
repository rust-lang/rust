//! Defines hir-level representation of impls.
//!
//! The handling is similar, but is not quite the same as for other items,
//! because `impl`s don't have names.

use std::sync::Arc;

use hir_expand::AstId;
use ra_syntax::ast;

use crate::{
    db::DefDatabase2, type_ref::TypeRef, AssocItemId, AstItemDef, ConstId, FunctionContainerId,
    FunctionLoc, ImplId, Intern, LocationCtx, TypeAliasId,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImplData {
    target_trait: Option<TypeRef>,
    target_type: TypeRef,
    items: Vec<AssocItemId>,
    negative: bool,
}

impl ImplData {
    pub(crate) fn impl_data_query(db: &impl DefDatabase2, id: ImplId) -> Arc<ImplData> {
        let src = id.source(db);
        let items = db.ast_id_map(src.file_id);

        let target_trait = src.value.target_trait().map(TypeRef::from_ast);
        let target_type = TypeRef::from_ast_opt(src.value.target_type());
        let negative = src.value.is_negative();

        let items = if let Some(item_list) = src.value.item_list() {
            let ctx = LocationCtx::new(db, id.module(db), src.file_id);
            item_list
                .impl_items()
                .map(|item_node| match item_node {
                    ast::ImplItem::FnDef(it) => {
                        let func_id = FunctionLoc {
                            container: FunctionContainerId::ImplId(id),
                            ast_id: AstId::new(src.file_id, items.ast_id(&it)),
                        }
                        .intern(db);
                        func_id.into()
                    }
                    ast::ImplItem::ConstDef(it) => {
                        ConstId::from_ast_id(ctx, items.ast_id(&it)).into()
                    }
                    ast::ImplItem::TypeAliasDef(it) => {
                        TypeAliasId::from_ast_id(ctx, items.ast_id(&it)).into()
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

        let res = ImplData { target_trait, target_type, items, negative };
        Arc::new(res)
    }

    pub fn target_trait(&self) -> Option<&TypeRef> {
        self.target_trait.as_ref()
    }

    pub fn target_type(&self) -> &TypeRef {
        &self.target_type
    }

    pub fn items(&self) -> &[AssocItemId] {
        &self.items
    }

    pub fn is_negative(&self) -> bool {
        self.negative
    }
}
