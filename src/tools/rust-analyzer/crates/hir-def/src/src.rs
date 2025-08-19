//! Utilities for mapping between hir IDs and the surface syntax.

use either::Either;
use hir_expand::{AstId, InFile};
use la_arena::{Arena, ArenaMap, Idx};
use syntax::{AstNode, AstPtr, ast};

use crate::{
    AstIdLoc, GenericDefId, LocalFieldId, LocalLifetimeParamId, LocalTypeOrConstParamId, Lookup,
    UseId, VariantId, attr::Attrs, db::DefDatabase,
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
    T: AstIdLoc,
{
    type Value = T::Ast;
    fn ast_ptr(&self, db: &dyn DefDatabase) -> InFile<AstPtr<Self::Value>> {
        let id = self.ast_id();
        let ast_id_map = db.ast_id_map(id.file_id);
        InFile::new(id.file_id, ast_id_map.get(id.value))
    }
}

pub trait HasChildSource<ChildId> {
    type Value;
    fn child_source(&self, db: &dyn DefDatabase) -> InFile<ArenaMap<ChildId, Self::Value>>;
}

/// Maps a `UseTree` contained in this import back to its AST node.
pub fn use_tree_to_ast(
    db: &dyn DefDatabase,
    use_ast_id: AstId<ast::Use>,
    index: Idx<ast::UseTree>,
) -> ast::UseTree {
    use_tree_source_map(db, use_ast_id)[index].clone()
}

/// Maps a `UseTree` contained in this import back to its AST node.
fn use_tree_source_map(db: &dyn DefDatabase, use_ast_id: AstId<ast::Use>) -> Arena<ast::UseTree> {
    // Re-lower the AST item and get the source map.
    // Note: The AST unwraps are fine, since if they fail we should have never obtained `index`.
    let ast = use_ast_id.to_node(db);
    let ast_use_tree = ast.use_tree().expect("missing `use_tree`");
    let mut span_map = None;
    crate::item_tree::lower_use_tree(db, ast_use_tree, &mut |range| {
        span_map.get_or_insert_with(|| db.span_map(use_ast_id.file_id)).span_for_range(range).ctx
    })
    .expect("failed to lower use tree")
    .1
}

impl HasChildSource<la_arena::Idx<ast::UseTree>> for UseId {
    type Value = ast::UseTree;
    fn child_source(
        &self,
        db: &dyn DefDatabase,
    ) -> InFile<ArenaMap<la_arena::Idx<ast::UseTree>, Self::Value>> {
        let loc = self.lookup(db);
        InFile::new(loc.id.file_id, use_tree_source_map(db, loc.id).into_iter().collect())
    }
}

impl HasChildSource<LocalTypeOrConstParamId> for GenericDefId {
    type Value = Either<ast::TypeOrConstParam, ast::TraitOrAlias>;
    fn child_source(
        &self,
        db: &dyn DefDatabase,
    ) -> InFile<ArenaMap<LocalTypeOrConstParamId, Self::Value>> {
        let generic_params = db.generic_params(*self);
        let mut idx_iter = generic_params.iter_type_or_consts().map(|(idx, _)| idx);

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
        let idx_iter = generic_params.iter_lt().map(|(idx, _)| idx);

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
        let (src, container) = match *self {
            VariantId::EnumVariantId(it) => {
                let lookup = it.lookup(db);
                (lookup.source(db).map(|it| it.kind()), lookup.parent.lookup(db).container)
            }
            VariantId::StructId(it) => {
                let lookup = it.lookup(db);
                (lookup.source(db).map(|it| it.kind()), lookup.container)
            }
            VariantId::UnionId(it) => {
                let lookup = it.lookup(db);
                (lookup.source(db).map(|it| it.kind()), lookup.container)
            }
        };
        let span_map = db.span_map(src.file_id);
        let mut map = ArenaMap::new();
        match &src.value {
            ast::StructKind::Tuple(fl) => {
                let cfg_options = container.krate.cfg_options(db);
                let mut idx = 0;
                for fd in fl.fields() {
                    let enabled =
                        Attrs::is_cfg_enabled_for(db, &fd, span_map.as_ref(), cfg_options).is_ok();
                    if !enabled {
                        continue;
                    }
                    map.insert(
                        LocalFieldId::from_raw(la_arena::RawIdx::from(idx)),
                        Either::Left(fd.clone()),
                    );
                    idx += 1;
                }
            }
            ast::StructKind::Record(fl) => {
                let cfg_options = container.krate.cfg_options(db);
                let mut idx = 0;
                for fd in fl.fields() {
                    let enabled =
                        Attrs::is_cfg_enabled_for(db, &fd, span_map.as_ref(), cfg_options).is_ok();
                    if !enabled {
                        continue;
                    }
                    map.insert(
                        LocalFieldId::from_raw(la_arena::RawIdx::from(idx)),
                        Either::Right(fd.clone()),
                    );
                    idx += 1;
                }
            }
            ast::StructKind::Unit => (),
        }
        InFile::new(src.file_id, map)
    }
}
