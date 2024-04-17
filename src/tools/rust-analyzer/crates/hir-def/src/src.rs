//! Utilities for mapping between hir IDs and the surface syntax.

use either::Either;
use hir_expand::InFile;
use la_arena::ArenaMap;
use syntax::{ast, AstNode, AstPtr};

use crate::{
    data::adt::lower_struct, db::DefDatabase, item_tree::ItemTreeNode, trace::Trace, GenericDefId,
    ItemTreeLoc, LocalFieldId, LocalLifetimeParamId, LocalTypeOrConstParamId, Lookup, UseId,
    VariantId,
};

pub trait HasSource {
    type Value: AstNode;
    fn source(&self, db: &dyn DefDatabase) -> InFile<Self::Value> {
        let InFile { file_id, value } = self.ast_ptr(db);
        InFile::new(file_id, value.to_node(&db.parse_or_expand(file_id)))
    }
    fn ast_ptr(&self, db: &dyn DefDatabase) -> InFile<AstPtr<Self::Value>>;
}

impl<T> HasSource for T
where
    T: ItemTreeLoc,
    T::Id: ItemTreeNode,
{
    type Value = <T::Id as ItemTreeNode>::Source;
    fn ast_ptr(&self, db: &dyn DefDatabase) -> InFile<AstPtr<Self::Value>> {
        let id = self.item_tree_id();
        let file_id = id.file_id();
        let tree = id.item_tree(db);
        let ast_id_map = db.ast_id_map(file_id);
        let node = &tree[id.value];

        InFile::new(file_id, ast_id_map.get(node.ast_id()))
    }
}

pub trait HasChildSource<ChildId> {
    type Value;
    fn child_source(&self, db: &dyn DefDatabase) -> InFile<ArenaMap<ChildId, Self::Value>>;
}

impl HasChildSource<la_arena::Idx<ast::UseTree>> for UseId {
    type Value = ast::UseTree;
    fn child_source(
        &self,
        db: &dyn DefDatabase,
    ) -> InFile<ArenaMap<la_arena::Idx<ast::UseTree>, Self::Value>> {
        let loc = &self.lookup(db);
        let use_ = &loc.id.item_tree(db)[loc.id.value];
        InFile::new(
            loc.id.file_id(),
            use_.use_tree_source_map(db, loc.id.file_id()).into_iter().collect(),
        )
    }
}

impl HasChildSource<LocalTypeOrConstParamId> for GenericDefId {
    type Value = Either<ast::TypeOrConstParam, ast::TraitOrAlias>;
    fn child_source(
        &self,
        db: &dyn DefDatabase,
    ) -> InFile<ArenaMap<LocalTypeOrConstParamId, Self::Value>> {
        let generic_params = db.generic_params(*self);
        let mut idx_iter = generic_params.type_or_consts.iter().map(|(idx, _)| idx);

        let (file_id, generic_params_list) = self.file_id_and_params_of(db);

        let mut params = ArenaMap::default();

        // For traits and trait aliases the first type index is `Self`, we need to add it before
        // the other params.
        match *self {
            GenericDefId::TraitId(id) => {
                let trait_ref = id.lookup(db).source(db).value;
                let idx = idx_iter.next().unwrap();
                params.insert(idx, Either::Right(ast::TraitOrAlias::Trait(trait_ref)));
            }
            GenericDefId::TraitAliasId(id) => {
                let alias = id.lookup(db).source(db).value;
                let idx = idx_iter.next().unwrap();
                params.insert(idx, Either::Right(ast::TraitOrAlias::TraitAlias(alias)));
            }
            _ => {}
        }

        if let Some(generic_params_list) = generic_params_list {
            for (idx, ast_param) in idx_iter.zip(generic_params_list.type_or_const_params()) {
                params.insert(idx, Either::Left(ast_param));
            }
        }

        InFile::new(file_id, params)
    }
}

impl HasChildSource<LocalLifetimeParamId> for GenericDefId {
    type Value = ast::LifetimeParam;
    fn child_source(
        &self,
        db: &dyn DefDatabase,
    ) -> InFile<ArenaMap<LocalLifetimeParamId, Self::Value>> {
        let generic_params = db.generic_params(*self);
        let idx_iter = generic_params.lifetimes.iter().map(|(idx, _)| idx);

        let (file_id, generic_params_list) = self.file_id_and_params_of(db);

        let mut params = ArenaMap::default();

        if let Some(generic_params_list) = generic_params_list {
            for (idx, ast_param) in idx_iter.zip(generic_params_list.lifetime_params()) {
                params.insert(idx, ast_param);
            }
        }

        InFile::new(file_id, params)
    }
}

impl HasChildSource<LocalFieldId> for VariantId {
    type Value = Either<ast::TupleField, ast::RecordField>;

    fn child_source(&self, db: &dyn DefDatabase) -> InFile<ArenaMap<LocalFieldId, Self::Value>> {
        let item_tree;
        let (src, fields, container) = match *self {
            VariantId::EnumVariantId(it) => {
                let lookup = it.lookup(db);
                item_tree = lookup.id.item_tree(db);
                (
                    lookup.source(db).map(|it| it.kind()),
                    &item_tree[lookup.id.value].fields,
                    lookup.parent.lookup(db).container,
                )
            }
            VariantId::StructId(it) => {
                let lookup = it.lookup(db);
                item_tree = lookup.id.item_tree(db);
                (
                    lookup.source(db).map(|it| it.kind()),
                    &item_tree[lookup.id.value].fields,
                    lookup.container,
                )
            }
            VariantId::UnionId(it) => {
                let lookup = it.lookup(db);
                item_tree = lookup.id.item_tree(db);
                (
                    lookup.source(db).map(|it| it.kind()),
                    &item_tree[lookup.id.value].fields,
                    lookup.container,
                )
            }
        };
        let mut trace = Trace::new_for_map();
        lower_struct(db, &mut trace, &src, container.krate, &item_tree, fields);
        src.with_value(trace.into_map())
    }
}
