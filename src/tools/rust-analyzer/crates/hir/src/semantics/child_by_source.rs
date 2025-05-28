//! When *constructing* `hir`, we start at some parent syntax node and recursively
//! lower the children.
//!
//! This module allows one to go in the opposite direction: start with a syntax
//! node for a *child*, and get its hir.

use either::Either;
use hir_expand::{HirFileId, attrs::collect_attrs};
use syntax::{AstPtr, ast};

use hir_def::{
    AdtId, AssocItemId, DefWithBodyId, EnumId, FieldId, GenericDefId, ImplId, ItemTreeLoc,
    LifetimeParamId, Lookup, MacroId, ModuleDefId, ModuleId, TraitId, TypeOrConstParamId,
    VariantId,
    db::DefDatabase,
    dyn_map::{
        DynMap,
        keys::{self, Key},
    },
    item_scope::ItemScope,
    item_tree::ItemTreeNode,
    nameres::DefMap,
    src::{HasChildSource, HasSource},
};

pub(crate) trait ChildBySource {
    fn child_by_source(&self, db: &dyn DefDatabase, file_id: HirFileId) -> DynMap {
        let mut res = DynMap::default();
        self.child_by_source_to(db, &mut res, file_id);
        res
    }
    fn child_by_source_to(&self, db: &dyn DefDatabase, map: &mut DynMap, file_id: HirFileId);
}

impl ChildBySource for TraitId {
    fn child_by_source_to(&self, db: &dyn DefDatabase, res: &mut DynMap, file_id: HirFileId) {
        let data = db.trait_items(*self);

        data.macro_calls().filter(|(ast_id, _)| ast_id.file_id == file_id).for_each(
            |(ast_id, call_id)| {
                let ptr = ast_id.to_ptr(db);
                if let Some(ptr) = ptr.cast::<ast::MacroCall>() {
                    res[keys::MACRO_CALL].insert(ptr, call_id);
                } else {
                    res[keys::ATTR_MACRO_CALL].insert(ptr, call_id);
                }
            },
        );
        data.items.iter().for_each(|&(_, item)| {
            add_assoc_item(db, res, file_id, item);
        });
    }
}

impl ChildBySource for ImplId {
    fn child_by_source_to(&self, db: &dyn DefDatabase, res: &mut DynMap, file_id: HirFileId) {
        let data = db.impl_items(*self);
        data.macro_calls().filter(|(ast_id, _)| ast_id.file_id == file_id).for_each(
            |(ast_id, call_id)| {
                let ptr = ast_id.to_ptr(db);
                if let Some(ptr) = ptr.cast::<ast::MacroCall>() {
                    res[keys::MACRO_CALL].insert(ptr, call_id);
                } else {
                    res[keys::ATTR_MACRO_CALL].insert(ptr, call_id);
                }
            },
        );
        data.items.iter().for_each(|&(_, item)| {
            add_assoc_item(db, res, file_id, item);
        });
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
        self.impls().for_each(|imp| insert_item_loc(db, res, file_id, imp, keys::IMPL));
        self.extern_blocks().for_each(|extern_block| {
            insert_item_loc(db, res, file_id, extern_block, keys::EXTERN_BLOCK)
        });
        self.extern_crate_decls()
            .for_each(|ext| insert_item_loc(db, res, file_id, ext, keys::EXTERN_CRATE));
        self.use_decls().for_each(|ext| insert_item_loc(db, res, file_id, ext, keys::USE));
        self.unnamed_consts()
            .for_each(|konst| insert_item_loc(db, res, file_id, konst, keys::CONST));
        self.attr_macro_invocs().filter(|(id, _)| id.file_id == file_id).for_each(
            |(ast_id, call_id)| {
                res[keys::ATTR_MACRO_CALL].insert(ast_id.to_ptr(db), call_id);
            },
        );
        self.legacy_macros().for_each(|(_, ids)| {
            ids.iter().for_each(|&id| {
                if let MacroId::MacroRulesId(id) = id {
                    let loc = id.lookup(db);
                    if loc.id.file_id() == file_id {
                        res[keys::MACRO_RULES].insert(loc.ast_ptr(db).value, id);
                    }
                }
            })
        });
        self.derive_macro_invocs().filter(|(id, _)| id.file_id == file_id).for_each(
            |(ast_id, calls)| {
                let adt = ast_id.to_node(db);
                calls.for_each(|(attr_id, call_id, calls)| {
                    if let Some((_, Either::Left(attr))) =
                        collect_attrs(&adt).nth(attr_id.ast_index())
                    {
                        res[keys::DERIVE_MACRO_CALL]
                            .insert(AstPtr::new(&attr), (attr_id, call_id, calls.into()));
                    }
                });
            },
        );
        self.iter_macro_invoc().filter(|(id, _)| id.file_id == file_id).for_each(
            |(ast_id, &call)| {
                let ast = ast_id.to_ptr(db);
                res[keys::MACRO_CALL].insert(ast, call);
            },
        );
        fn add_module_def(
            db: &dyn DefDatabase,
            map: &mut DynMap,
            file_id: HirFileId,
            item: ModuleDefId,
        ) {
            match item {
                ModuleDefId::FunctionId(id) => {
                    insert_item_loc(db, map, file_id, id, keys::FUNCTION)
                }
                ModuleDefId::ConstId(id) => insert_item_loc(db, map, file_id, id, keys::CONST),
                ModuleDefId::TypeAliasId(id) => {
                    insert_item_loc(db, map, file_id, id, keys::TYPE_ALIAS)
                }
                ModuleDefId::StaticId(id) => insert_item_loc(db, map, file_id, id, keys::STATIC),
                ModuleDefId::TraitId(id) => insert_item_loc(db, map, file_id, id, keys::TRAIT),
                ModuleDefId::TraitAliasId(id) => {
                    insert_item_loc(db, map, file_id, id, keys::TRAIT_ALIAS)
                }
                ModuleDefId::AdtId(adt) => match adt {
                    AdtId::StructId(id) => insert_item_loc(db, map, file_id, id, keys::STRUCT),
                    AdtId::UnionId(id) => insert_item_loc(db, map, file_id, id, keys::UNION),
                    AdtId::EnumId(id) => insert_item_loc(db, map, file_id, id, keys::ENUM),
                },
                ModuleDefId::MacroId(id) => match id {
                    MacroId::Macro2Id(id) => insert_item_loc(db, map, file_id, id, keys::MACRO2),
                    MacroId::MacroRulesId(id) => {
                        insert_item_loc(db, map, file_id, id, keys::MACRO_RULES)
                    }
                    MacroId::ProcMacroId(id) => {
                        insert_item_loc(db, map, file_id, id, keys::PROC_MACRO)
                    }
                },
                ModuleDefId::ModuleId(_)
                | ModuleDefId::EnumVariantId(_)
                | ModuleDefId::BuiltinType(_) => (),
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
                Either::Left(source) => res[keys::TUPLE_FIELD].insert(AstPtr::new(&source), id),
                Either::Right(source) => res[keys::RECORD_FIELD].insert(AstPtr::new(&source), id),
            }
        }
    }
}

impl ChildBySource for EnumId {
    fn child_by_source_to(&self, db: &dyn DefDatabase, res: &mut DynMap, file_id: HirFileId) {
        let loc = &self.lookup(db);
        if file_id != loc.id.file_id() {
            return;
        }

        let tree = loc.id.item_tree(db);
        let ast_id_map = db.ast_id_map(loc.id.file_id());

        db.enum_variants(*self).variants.iter().for_each(|&(variant, _)| {
            res[keys::ENUM_VARIANT]
                .insert(ast_id_map.get(tree[variant.lookup(db).id.value].ast_id), variant);
        });
    }
}

impl ChildBySource for DefWithBodyId {
    fn child_by_source_to(&self, db: &dyn DefDatabase, res: &mut DynMap, file_id: HirFileId) {
        let (body, sm) = db.body_with_source_map(*self);
        if let &DefWithBodyId::VariantId(v) = self {
            VariantId::EnumVariantId(v).child_by_source_to(db, res, file_id)
        }

        sm.expansions().filter(|(ast, _)| ast.file_id == file_id).for_each(|(ast, &exp_id)| {
            res[keys::MACRO_CALL].insert(ast.value, exp_id);
        });

        for (block, def_map) in body.blocks(db) {
            // All block expressions are merged into the same map, because they logically all add
            // inner items to the containing `DefWithBodyId`.
            def_map[DefMap::ROOT].scope.child_by_source_to(db, res, file_id);
            res[keys::BLOCK].insert(block.lookup(db).ast_id.to_ptr(db), block);
        }
    }
}

impl ChildBySource for GenericDefId {
    fn child_by_source_to(&self, db: &dyn DefDatabase, res: &mut DynMap, file_id: HirFileId) {
        let (gfile_id, generic_params_list) = self.file_id_and_params_of(db);
        if gfile_id != file_id {
            return;
        }

        let generic_params = db.generic_params(*self);
        let mut toc_idx_iter = generic_params.iter_type_or_consts().map(|(idx, _)| idx);
        let lts_idx_iter = generic_params.iter_lt().map(|(idx, _)| idx);

        // For traits the first type index is `Self`, skip it.
        if let GenericDefId::TraitId(_) = *self {
            toc_idx_iter.next().unwrap(); // advance_by(1);
        }

        if let Some(generic_params_list) = generic_params_list {
            for (local_id, ast_param) in
                toc_idx_iter.zip(generic_params_list.type_or_const_params())
            {
                let id = TypeOrConstParamId { parent: *self, local_id };
                match ast_param {
                    ast::TypeOrConstParam::Type(a) => {
                        res[keys::TYPE_PARAM].insert(AstPtr::new(&a), id)
                    }
                    ast::TypeOrConstParam::Const(a) => {
                        res[keys::CONST_PARAM].insert(AstPtr::new(&a), id)
                    }
                }
            }
            for (local_id, ast_param) in lts_idx_iter.zip(generic_params_list.lifetime_params()) {
                let id = LifetimeParamId { parent: *self, local_id };
                res[keys::LIFETIME_PARAM].insert(AstPtr::new(&ast_param), id);
            }
        }
    }
}

fn insert_item_loc<ID, N, Data>(
    db: &dyn DefDatabase,
    res: &mut DynMap,
    file_id: HirFileId,
    id: ID,
    key: Key<N::Source, ID>,
) where
    ID: Lookup<Database = dyn DefDatabase, Data = Data> + 'static,
    Data: ItemTreeLoc<Id = N>,
    N: ItemTreeNode,
    N::Source: 'static,
{
    let loc = id.lookup(db);
    if loc.item_tree_id().file_id() == file_id {
        res[key].insert(loc.ast_ptr(db).value, id)
    }
}

fn add_assoc_item(db: &dyn DefDatabase, res: &mut DynMap, file_id: HirFileId, item: AssocItemId) {
    match item {
        AssocItemId::FunctionId(func) => insert_item_loc(db, res, file_id, func, keys::FUNCTION),
        AssocItemId::ConstId(konst) => insert_item_loc(db, res, file_id, konst, keys::CONST),
        AssocItemId::TypeAliasId(ty) => insert_item_loc(db, res, file_id, ty, keys::TYPE_ALIAS),
    }
}
