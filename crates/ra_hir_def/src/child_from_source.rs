//! When *constructing* `hir`, we start at some parent syntax node and recursively
//! lower the children.
//!
//! This modules allows one to go in the opposite direction: start with a syntax
//! node for a *child*, and get its hir.

use either::Either;
use hir_expand::InFile;
use ra_syntax::{ast, AstNode, AstPtr};

use crate::{
    db::DefDatabase,
    src::{HasChildSource, HasSource},
    AssocItemId, ConstId, EnumId, EnumVariantId, FunctionId, ImplId, Lookup, ModuleDefId, ModuleId,
    StaticId, StructFieldId, TraitId, TypeAliasId, VariantId,
};

pub trait ChildFromSource<CHILD, SOURCE> {
    fn child_from_source(
        &self,
        db: &impl DefDatabase,
        child_source: InFile<SOURCE>,
    ) -> Option<CHILD>;
}

impl ChildFromSource<FunctionId, ast::FnDef> for TraitId {
    fn child_from_source(
        &self,
        db: &impl DefDatabase,
        child_source: InFile<ast::FnDef>,
    ) -> Option<FunctionId> {
        let data = db.trait_data(*self);
        data.items
            .iter()
            .filter_map(|(_, item)| match item {
                AssocItemId::FunctionId(it) => Some(*it),
                _ => None,
            })
            .find(|func| {
                let source = func.lookup(db).source(db);
                same_source(&source, &child_source)
            })
    }
}

impl ChildFromSource<FunctionId, ast::FnDef> for ImplId {
    fn child_from_source(
        &self,
        db: &impl DefDatabase,
        child_source: InFile<ast::FnDef>,
    ) -> Option<FunctionId> {
        let data = db.impl_data(*self);
        data.items
            .iter()
            .filter_map(|item| match item {
                AssocItemId::FunctionId(it) => Some(*it),
                _ => None,
            })
            .find(|func| {
                let source = func.lookup(db).source(db);
                same_source(&source, &child_source)
            })
    }
}

impl ChildFromSource<FunctionId, ast::FnDef> for ModuleId {
    fn child_from_source(
        &self,
        db: &impl DefDatabase,
        child_source: InFile<ast::FnDef>,
    ) -> Option<FunctionId> {
        let crate_def_map = db.crate_def_map(self.krate);
        let res = crate_def_map[self.local_id]
            .scope
            .declarations()
            .filter_map(|item| match item {
                ModuleDefId::FunctionId(it) => Some(it),
                _ => None,
            })
            .find(|func| {
                let source = func.lookup(db).source(db);
                same_source(&source, &child_source)
            });
        res
    }
}

impl ChildFromSource<ConstId, ast::ConstDef> for TraitId {
    fn child_from_source(
        &self,
        db: &impl DefDatabase,
        child_source: InFile<ast::ConstDef>,
    ) -> Option<ConstId> {
        let data = db.trait_data(*self);
        data.items
            .iter()
            .filter_map(|(_, item)| match item {
                AssocItemId::ConstId(it) => Some(*it),
                _ => None,
            })
            .find(|func| {
                let source = func.lookup(db).source(db);
                same_source(&source, &child_source)
            })
    }
}

impl ChildFromSource<ConstId, ast::ConstDef> for ImplId {
    fn child_from_source(
        &self,
        db: &impl DefDatabase,
        child_source: InFile<ast::ConstDef>,
    ) -> Option<ConstId> {
        let data = db.impl_data(*self);
        data.items
            .iter()
            .filter_map(|item| match item {
                AssocItemId::ConstId(it) => Some(*it),
                _ => None,
            })
            .find(|func| {
                let source = func.lookup(db).source(db);
                same_source(&source, &child_source)
            })
    }
}

impl ChildFromSource<ConstId, ast::ConstDef> for ModuleId {
    fn child_from_source(
        &self,
        db: &impl DefDatabase,
        child_source: InFile<ast::ConstDef>,
    ) -> Option<ConstId> {
        let crate_def_map = db.crate_def_map(self.krate);
        let res = crate_def_map[self.local_id]
            .scope
            .declarations()
            .filter_map(|item| match item {
                ModuleDefId::ConstId(it) => Some(it),
                _ => None,
            })
            .find(|func| {
                let source = func.lookup(db).source(db);
                same_source(&source, &child_source)
            });
        res
    }
}

impl ChildFromSource<TypeAliasId, ast::TypeAliasDef> for TraitId {
    fn child_from_source(
        &self,
        db: &impl DefDatabase,
        child_source: InFile<ast::TypeAliasDef>,
    ) -> Option<TypeAliasId> {
        let data = db.trait_data(*self);
        data.items
            .iter()
            .filter_map(|(_, item)| match item {
                AssocItemId::TypeAliasId(it) => Some(*it),
                _ => None,
            })
            .find(|func| {
                let source = func.lookup(db).source(db);
                same_source(&source, &child_source)
            })
    }
}

impl ChildFromSource<TypeAliasId, ast::TypeAliasDef> for ImplId {
    fn child_from_source(
        &self,
        db: &impl DefDatabase,
        child_source: InFile<ast::TypeAliasDef>,
    ) -> Option<TypeAliasId> {
        let data = db.impl_data(*self);
        data.items
            .iter()
            .filter_map(|item| match item {
                AssocItemId::TypeAliasId(it) => Some(*it),
                _ => None,
            })
            .find(|func| {
                let source = func.lookup(db).source(db);
                same_source(&source, &child_source)
            })
    }
}

impl ChildFromSource<TypeAliasId, ast::TypeAliasDef> for ModuleId {
    fn child_from_source(
        &self,
        db: &impl DefDatabase,
        child_source: InFile<ast::TypeAliasDef>,
    ) -> Option<TypeAliasId> {
        let crate_def_map = db.crate_def_map(self.krate);
        let res = crate_def_map[self.local_id]
            .scope
            .declarations()
            .filter_map(|item| match item {
                ModuleDefId::TypeAliasId(it) => Some(it),
                _ => None,
            })
            .find(|func| {
                let source = func.lookup(db).source(db);
                same_source(&source, &child_source)
            });
        res
    }
}

impl ChildFromSource<StaticId, ast::StaticDef> for ModuleId {
    fn child_from_source(
        &self,
        db: &impl DefDatabase,
        child_source: InFile<ast::StaticDef>,
    ) -> Option<StaticId> {
        let crate_def_map = db.crate_def_map(self.krate);
        let res = crate_def_map[self.local_id]
            .scope
            .declarations()
            .filter_map(|item| match item {
                ModuleDefId::StaticId(it) => Some(it),
                _ => None,
            })
            .find(|func| {
                let source = func.lookup(db).source(db);
                same_source(&source, &child_source)
            });
        res
    }
}

impl ChildFromSource<StructFieldId, Either<ast::TupleFieldDef, ast::RecordFieldDef>> for VariantId {
    fn child_from_source(
        &self,
        db: &impl DefDatabase,
        child_source: InFile<Either<ast::TupleFieldDef, ast::RecordFieldDef>>,
    ) -> Option<StructFieldId> {
        let arena_map = self.child_source(db);
        let (local_id, _) = arena_map.as_ref().value.iter().find(|(_local_id, source)| {
            child_source.file_id == arena_map.file_id
                && match (source, &child_source.value) {
                    (Either::Left(a), Either::Left(b)) => AstPtr::new(a) == AstPtr::new(b),
                    (Either::Right(a), Either::Right(b)) => AstPtr::new(a) == AstPtr::new(b),
                    _ => false,
                }
        })?;
        Some(StructFieldId { parent: *self, local_id })
    }
}

impl ChildFromSource<EnumVariantId, ast::EnumVariant> for EnumId {
    fn child_from_source(
        &self,
        db: &impl DefDatabase,
        child_source: InFile<ast::EnumVariant>,
    ) -> Option<EnumVariantId> {
        let arena_map = self.child_source(db);
        let (local_id, _) = arena_map.as_ref().value.iter().find(|(_local_id, source)| {
            child_source.file_id == arena_map.file_id
                && AstPtr::new(*source) == AstPtr::new(&child_source.value)
        })?;
        Some(EnumVariantId { parent: *self, local_id })
    }
}

/// XXX: AST Nodes and SyntaxNodes have identity equality semantics: nodes are
/// equal if they point to exactly the same object.
///
/// In general, we do not guarantee that we have exactly one instance of a
/// syntax tree for each file. We probably should add such guarantee, but, for
/// the time being, we will use identity-less AstPtr comparison.
fn same_source<N: AstNode>(s1: &InFile<N>, s2: &InFile<N>) -> bool {
    s1.as_ref().map(AstPtr::new) == s2.as_ref().map(AstPtr::new)
}
