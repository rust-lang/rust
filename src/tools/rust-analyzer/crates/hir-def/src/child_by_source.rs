//! When *constructing* `hir`, we start at some parent syntax node and recursively
//! lower the children.
//!
//! This modules allows one to go in the opposite direction: start with a syntax
//! node for a *child*, and get its hir.

use either::Either;
use hir_expand::HirFileId;
use syntax::ast::HasDocComments;

use crate::{
    db::DefDatabase,
    dyn_map::DynMap,
    item_scope::ItemScope,
    keys,
    src::{HasChildSource, HasSource},
    AdtId, AssocItemId, DefWithBodyId, EnumId, EnumVariantId, FieldId, ImplId, Lookup, MacroId,
    ModuleDefId, ModuleId, TraitId, VariantId,
};

pub trait ChildBySource {
    fn child_by_source(&self, db: &dyn DefDatabase, file_id: HirFileId) -> DynMap {
        let mut res = DynMap::default();
        self.child_by_source_to(db, &mut res, file_id);
        res
    }
    fn child_by_source_to(&self, db: &dyn DefDatabase, map: &mut DynMap, file_id: HirFileId);
}

impl ChildBySource for TraitId {
    fn child_by_source_to(&self, db: &dyn DefDatabase, res: &mut DynMap, file_id: HirFileId) {
        let data = db.trait_data(*self);

        data.attribute_calls().filter(|(ast_id, _)| ast_id.file_id == file_id).for_each(
            |(ast_id, call_id)| {
                res[keys::ATTR_MACRO_CALL].insert(ast_id.to_node(db.upcast()), call_id);
            },
        );
        data.items.iter().for_each(|&(_, item)| {
            add_assoc_item(db, res, file_id, item);
        });
    }
}

impl ChildBySource for ImplId {
    fn child_by_source_to(&self, db: &dyn DefDatabase, res: &mut DynMap, file_id: HirFileId) {
        let data = db.impl_data(*self);
        data.attribute_calls().filter(|(ast_id, _)| ast_id.file_id == file_id).for_each(
            |(ast_id, call_id)| {
                res[keys::ATTR_MACRO_CALL].insert(ast_id.to_node(db.upcast()), call_id);
            },
        );
        data.items.iter().for_each(|&item| {
            add_assoc_item(db, res, file_id, item);
        });
    }
}

fn add_assoc_item(db: &dyn DefDatabase, res: &mut DynMap, file_id: HirFileId, item: AssocItemId) {
    match item {
        AssocItemId::FunctionId(func) => {
            let loc = func.lookup(db);
            if loc.id.file_id() == file_id {
                res[keys::FUNCTION].insert(loc.source(db).value, func)
            }
        }
        AssocItemId::ConstId(konst) => {
            let loc = konst.lookup(db);
            if loc.id.file_id() == file_id {
                res[keys::CONST].insert(loc.source(db).value, konst)
            }
        }
        AssocItemId::TypeAliasId(ty) => {
            let loc = ty.lookup(db);
            if loc.id.file_id() == file_id {
                res[keys::TYPE_ALIAS].insert(loc.source(db).value, ty)
            }
        }
    }
}

impl ChildBySource for ModuleId {
    fn child_by_source_to(&self, db: &dyn DefDatabase, res: &mut DynMap, file_id: HirFileId) {
        let def_map = self.def_map(db);
        let module_data = &def_map[self.local_id];
        module_data.scope.child_by_source_to(db, res, file_id);
    }
}

impl ChildBySource for ItemScope {
    fn child_by_source_to(&self, db: &dyn DefDatabase, res: &mut DynMap, file_id: HirFileId) {
        self.declarations().for_each(|item| add_module_def(db, res, file_id, item));
        self.impls().for_each(|imp| add_impl(db, res, file_id, imp));
        self.unnamed_consts().for_each(|konst| {
            let loc = konst.lookup(db);
            if loc.id.file_id() == file_id {
                res[keys::CONST].insert(loc.source(db).value, konst);
            }
        });
        self.attr_macro_invocs().filter(|(id, _)| id.file_id == file_id).for_each(
            |(ast_id, call_id)| {
                res[keys::ATTR_MACRO_CALL].insert(ast_id.to_node(db.upcast()), call_id);
            },
        );
        self.legacy_macros().for_each(|(_, ids)| {
            ids.iter().for_each(|&id| {
                if let MacroId::MacroRulesId(id) = id {
                    let loc = id.lookup(db);
                    if loc.id.file_id() == file_id {
                        res[keys::MACRO_RULES].insert(loc.source(db).value, id);
                    }
                }
            })
        });
        self.derive_macro_invocs().filter(|(id, _)| id.file_id == file_id).for_each(
            |(ast_id, calls)| {
                let adt = ast_id.to_node(db.upcast());
                calls.for_each(|(attr_id, call_id, calls)| {
                    if let Some(Either::Left(attr)) =
                        adt.doc_comments_and_attrs().nth(attr_id.ast_index())
                    {
                        res[keys::DERIVE_MACRO_CALL].insert(attr, (attr_id, call_id, calls.into()));
                    }
                });
            },
        );

        fn add_module_def(
            db: &dyn DefDatabase,
            map: &mut DynMap,
            file_id: HirFileId,
            item: ModuleDefId,
        ) {
            macro_rules! insert {
                ($map:ident[$key:path].$insert:ident($id:ident)) => {{
                    let loc = $id.lookup(db);
                    if loc.id.file_id() == file_id {
                        $map[$key].$insert(loc.source(db).value, $id)
                    }
                }};
            }
            match item {
                ModuleDefId::FunctionId(id) => insert!(map[keys::FUNCTION].insert(id)),
                ModuleDefId::ConstId(id) => insert!(map[keys::CONST].insert(id)),
                ModuleDefId::StaticId(id) => insert!(map[keys::STATIC].insert(id)),
                ModuleDefId::TypeAliasId(id) => insert!(map[keys::TYPE_ALIAS].insert(id)),
                ModuleDefId::TraitId(id) => insert!(map[keys::TRAIT].insert(id)),
                ModuleDefId::AdtId(adt) => match adt {
                    AdtId::StructId(id) => insert!(map[keys::STRUCT].insert(id)),
                    AdtId::UnionId(id) => insert!(map[keys::UNION].insert(id)),
                    AdtId::EnumId(id) => insert!(map[keys::ENUM].insert(id)),
                },
                ModuleDefId::MacroId(id) => match id {
                    MacroId::Macro2Id(id) => insert!(map[keys::MACRO2].insert(id)),
                    MacroId::MacroRulesId(id) => insert!(map[keys::MACRO_RULES].insert(id)),
                    MacroId::ProcMacroId(id) => insert!(map[keys::PROC_MACRO].insert(id)),
                },
                ModuleDefId::ModuleId(_)
                | ModuleDefId::EnumVariantId(_)
                | ModuleDefId::BuiltinType(_) => (),
            }
        }
        fn add_impl(db: &dyn DefDatabase, map: &mut DynMap, file_id: HirFileId, imp: ImplId) {
            let loc = imp.lookup(db);
            if loc.id.file_id() == file_id {
                map[keys::IMPL].insert(loc.source(db).value, imp)
            }
        }
    }
}

impl ChildBySource for VariantId {
    fn child_by_source_to(&self, db: &dyn DefDatabase, res: &mut DynMap, _: HirFileId) {
        let arena_map = self.child_source(db);
        let arena_map = arena_map.as_ref();
        let parent = *self;
        for (local_id, source) in arena_map.value.iter() {
            let id = FieldId { parent, local_id };
            match source.clone() {
                Either::Left(source) => res[keys::TUPLE_FIELD].insert(source, id),
                Either::Right(source) => res[keys::RECORD_FIELD].insert(source, id),
            }
        }
    }
}

impl ChildBySource for EnumId {
    fn child_by_source_to(&self, db: &dyn DefDatabase, res: &mut DynMap, _: HirFileId) {
        let arena_map = self.child_source(db);
        let arena_map = arena_map.as_ref();
        for (local_id, source) in arena_map.value.iter() {
            let id = EnumVariantId { parent: *self, local_id };
            res[keys::VARIANT].insert(source.clone(), id)
        }
    }
}

impl ChildBySource for DefWithBodyId {
    fn child_by_source_to(&self, db: &dyn DefDatabase, res: &mut DynMap, file_id: HirFileId) {
        let body = db.body(*self);
        if let &DefWithBodyId::VariantId(v) = self {
            VariantId::EnumVariantId(v).child_by_source_to(db, res, file_id)
        }

        for (_, def_map) in body.blocks(db) {
            // All block expressions are merged into the same map, because they logically all add
            // inner items to the containing `DefWithBodyId`.
            def_map[def_map.root()].scope.child_by_source_to(db, res, file_id);
        }
    }
}
