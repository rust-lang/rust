//! FIXME: write short doc here

use hir_def::{type_ref::TypeRef, AstItemDef};
use ra_syntax::ast::{self};

use crate::{
    db::{AstDatabase, DefDatabase, HirDatabase},
    generics::HasGenericParams,
    resolve::Resolver,
    ty::Ty,
    AssocItem, Crate, HasSource, ImplBlock, Module, Source, TraitRef,
};

impl HasSource for ImplBlock {
    type Ast = ast::ImplBlock;
    fn source(self, db: &(impl DefDatabase + AstDatabase)) -> Source<ast::ImplBlock> {
        self.id.source(db)
    }
}

impl ImplBlock {
    pub(crate) fn containing(db: &impl DefDatabase, item: AssocItem) -> Option<ImplBlock> {
        let module = item.module(db);
        let crate_def_map = db.crate_def_map(module.id.krate);
        crate_def_map[module.id.module_id].impls.iter().copied().map(ImplBlock::from).find(|it| {
            db.impl_data(it.id).items().iter().copied().map(AssocItem::from).any(|it| it == item)
        })
    }

    pub fn target_trait(&self, db: &impl DefDatabase) -> Option<TypeRef> {
        db.impl_data(self.id).target_trait().cloned()
    }

    pub fn target_type(&self, db: &impl DefDatabase) -> TypeRef {
        db.impl_data(self.id).target_type().clone()
    }

    pub fn target_ty(&self, db: &impl HirDatabase) -> Ty {
        Ty::from_hir(db, &self.resolver(db), &self.target_type(db))
    }

    pub fn target_trait_ref(&self, db: &impl HirDatabase) -> Option<TraitRef> {
        let target_ty = self.target_ty(db);
        TraitRef::from_hir(db, &self.resolver(db), &self.target_trait(db)?, Some(target_ty))
    }

    pub fn items(&self, db: &impl DefDatabase) -> Vec<AssocItem> {
        db.impl_data(self.id).items().iter().map(|it| (*it).into()).collect()
    }

    pub fn is_negative(&self, db: &impl DefDatabase) -> bool {
        db.impl_data(self.id).is_negative()
    }

    pub fn module(&self, db: &impl DefDatabase) -> Module {
        self.id.module(db).into()
    }

    pub fn krate(&self, db: &impl DefDatabase) -> Crate {
        Crate { crate_id: self.module(db).id.krate }
    }

    pub(crate) fn resolver(&self, db: &impl DefDatabase) -> Resolver {
        let r = self.module(db).resolver(db);
        // add generic params, if present
        let p = self.generic_params(db);
        let r = if !p.params.is_empty() { r.push_generic_params_scope(p) } else { r };
        let r = r.push_impl_block_scope(self.clone());
        r
    }
}
