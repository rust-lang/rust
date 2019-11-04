//! `hir_def` crate contains everything between macro expansion and type
//! inference.
//!
//! It defines various items (structs, enums, traits) which comprises Rust code,
//! as well as an algorithm for resolving paths to such entities.
//!
//! Note that `hir_def` is a work in progress, so not all of the above is
//! actually true.

pub mod db;
pub mod attr;
pub mod path;
pub mod type_ref;
pub mod builtin_type;
pub mod adt;
pub mod diagnostics;

#[cfg(test)]
mod test_db;
#[cfg(test)]
mod marks;

// FIXME: this should be private
pub mod nameres;

use std::hash::{Hash, Hasher};

use hir_expand::{ast_id_map::FileAstId, db::AstDatabase, AstId, HirFileId, Source};
use ra_arena::{impl_arena_id, RawId};
use ra_db::{salsa, CrateId, FileId};
use ra_syntax::{ast, AstNode, SyntaxNode};

use crate::{builtin_type::BuiltinType, db::InternDatabase};

pub enum ModuleSource {
    SourceFile(ast::SourceFile),
    Module(ast::Module),
}

impl ModuleSource {
    pub fn new(
        db: &impl db::DefDatabase2,
        file_id: Option<FileId>,
        decl_id: Option<AstId<ast::Module>>,
    ) -> ModuleSource {
        match (file_id, decl_id) {
            (Some(file_id), _) => {
                let source_file = db.parse(file_id).tree();
                ModuleSource::SourceFile(source_file)
            }
            (None, Some(item_id)) => {
                let module = item_id.to_node(db);
                assert!(module.item_list().is_some(), "expected inline module");
                ModuleSource::Module(module)
            }
            (None, None) => panic!(),
        }
    }

    // FIXME: this methods do not belong here
    pub fn from_position(
        db: &impl db::DefDatabase2,
        position: ra_db::FilePosition,
    ) -> ModuleSource {
        let parse = db.parse(position.file_id);
        match &ra_syntax::algo::find_node_at_offset::<ast::Module>(
            parse.tree().syntax(),
            position.offset,
        ) {
            Some(m) if !m.has_semi() => ModuleSource::Module(m.clone()),
            _ => {
                let source_file = parse.tree();
                ModuleSource::SourceFile(source_file)
            }
        }
    }

    pub fn from_child_node(
        db: &impl db::DefDatabase2,
        file_id: FileId,
        child: &SyntaxNode,
    ) -> ModuleSource {
        if let Some(m) = child.ancestors().filter_map(ast::Module::cast).find(|it| !it.has_semi()) {
            ModuleSource::Module(m)
        } else {
            let source_file = db.parse(file_id).tree();
            ModuleSource::SourceFile(source_file)
        }
    }

    pub fn from_file_id(db: &impl db::DefDatabase2, file_id: FileId) -> ModuleSource {
        let source_file = db.parse(file_id).tree();
        ModuleSource::SourceFile(source_file)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModuleId {
    pub krate: CrateId,
    pub module_id: CrateModuleId,
}

/// An ID of a module, **local** to a specific crate
// FIXME: rename to `LocalModuleId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CrateModuleId(RawId);
impl_arena_id!(CrateModuleId);

macro_rules! impl_intern_key {
    ($name:ident) => {
        impl salsa::InternKey for $name {
            fn from_intern_id(v: salsa::InternId) -> Self {
                $name(v)
            }
            fn as_intern_id(&self) -> salsa::InternId {
                self.0
            }
        }
    };
}

#[derive(Debug)]
pub struct ItemLoc<N: AstNode> {
    pub(crate) module: ModuleId,
    ast_id: AstId<N>,
}

impl<N: AstNode> PartialEq for ItemLoc<N> {
    fn eq(&self, other: &Self) -> bool {
        self.module == other.module && self.ast_id == other.ast_id
    }
}
impl<N: AstNode> Eq for ItemLoc<N> {}
impl<N: AstNode> Hash for ItemLoc<N> {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.module.hash(hasher);
        self.ast_id.hash(hasher);
    }
}

impl<N: AstNode> Clone for ItemLoc<N> {
    fn clone(&self) -> ItemLoc<N> {
        ItemLoc { module: self.module, ast_id: self.ast_id }
    }
}

#[derive(Clone, Copy)]
pub struct LocationCtx<DB> {
    db: DB,
    module: ModuleId,
    file_id: HirFileId,
}

impl<'a, DB> LocationCtx<&'a DB> {
    pub fn new(db: &'a DB, module: ModuleId, file_id: HirFileId) -> LocationCtx<&'a DB> {
        LocationCtx { db, module, file_id }
    }
}

impl<'a, DB: AstDatabase + InternDatabase> LocationCtx<&'a DB> {
    pub fn to_def<N, DEF>(self, ast: &N) -> DEF
    where
        N: AstNode,
        DEF: AstItemDef<N>,
    {
        DEF::from_ast(self, ast)
    }
}

pub trait AstItemDef<N: AstNode>: salsa::InternKey + Clone {
    fn intern(db: &impl InternDatabase, loc: ItemLoc<N>) -> Self;
    fn lookup_intern(self, db: &impl InternDatabase) -> ItemLoc<N>;

    fn from_ast(ctx: LocationCtx<&(impl AstDatabase + InternDatabase)>, ast: &N) -> Self {
        let items = ctx.db.ast_id_map(ctx.file_id);
        let item_id = items.ast_id(ast);
        Self::from_ast_id(ctx, item_id)
    }
    fn from_ast_id(ctx: LocationCtx<&impl InternDatabase>, ast_id: FileAstId<N>) -> Self {
        let loc = ItemLoc { module: ctx.module, ast_id: AstId::new(ctx.file_id, ast_id) };
        Self::intern(ctx.db, loc)
    }
    fn source(self, db: &(impl AstDatabase + InternDatabase)) -> Source<N> {
        let loc = self.lookup_intern(db);
        let ast = loc.ast_id.to_node(db);
        Source { file_id: loc.ast_id.file_id(), ast }
    }
    fn module(self, db: &impl InternDatabase) -> ModuleId {
        let loc = self.lookup_intern(db);
        loc.module
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FunctionId(salsa::InternId);
impl_intern_key!(FunctionId);

impl AstItemDef<ast::FnDef> for FunctionId {
    fn intern(db: &impl InternDatabase, loc: ItemLoc<ast::FnDef>) -> Self {
        db.intern_function(loc)
    }
    fn lookup_intern(self, db: &impl InternDatabase) -> ItemLoc<ast::FnDef> {
        db.lookup_intern_function(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StructId(salsa::InternId);
impl_intern_key!(StructId);
impl AstItemDef<ast::StructDef> for StructId {
    fn intern(db: &impl InternDatabase, loc: ItemLoc<ast::StructDef>) -> Self {
        db.intern_struct(loc)
    }
    fn lookup_intern(self, db: &impl InternDatabase) -> ItemLoc<ast::StructDef> {
        db.lookup_intern_struct(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnionId(salsa::InternId);
impl_intern_key!(UnionId);
impl AstItemDef<ast::StructDef> for UnionId {
    fn intern(db: &impl InternDatabase, loc: ItemLoc<ast::StructDef>) -> Self {
        db.intern_union(loc)
    }
    fn lookup_intern(self, db: &impl InternDatabase) -> ItemLoc<ast::StructDef> {
        db.lookup_intern_union(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EnumId(salsa::InternId);
impl_intern_key!(EnumId);
impl AstItemDef<ast::EnumDef> for EnumId {
    fn intern(db: &impl InternDatabase, loc: ItemLoc<ast::EnumDef>) -> Self {
        db.intern_enum(loc)
    }
    fn lookup_intern(self, db: &impl InternDatabase) -> ItemLoc<ast::EnumDef> {
        db.lookup_intern_enum(self)
    }
}

// FIXME: rename to `VariantId`, only enums can ave variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EnumVariantId {
    pub parent: EnumId,
    pub local_id: LocalEnumVariantId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LocalEnumVariantId(RawId);
impl_arena_id!(LocalEnumVariantId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VariantId {
    EnumVariantId(EnumVariantId),
    StructId(StructId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StructFieldId {
    parent: VariantId,
    local_id: LocalStructFieldId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LocalStructFieldId(RawId);
impl_arena_id!(LocalStructFieldId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConstId(salsa::InternId);
impl_intern_key!(ConstId);
impl AstItemDef<ast::ConstDef> for ConstId {
    fn intern(db: &impl InternDatabase, loc: ItemLoc<ast::ConstDef>) -> Self {
        db.intern_const(loc)
    }
    fn lookup_intern(self, db: &impl InternDatabase) -> ItemLoc<ast::ConstDef> {
        db.lookup_intern_const(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StaticId(salsa::InternId);
impl_intern_key!(StaticId);
impl AstItemDef<ast::StaticDef> for StaticId {
    fn intern(db: &impl InternDatabase, loc: ItemLoc<ast::StaticDef>) -> Self {
        db.intern_static(loc)
    }
    fn lookup_intern(self, db: &impl InternDatabase) -> ItemLoc<ast::StaticDef> {
        db.lookup_intern_static(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TraitId(salsa::InternId);
impl_intern_key!(TraitId);
impl AstItemDef<ast::TraitDef> for TraitId {
    fn intern(db: &impl InternDatabase, loc: ItemLoc<ast::TraitDef>) -> Self {
        db.intern_trait(loc)
    }
    fn lookup_intern(self, db: &impl InternDatabase) -> ItemLoc<ast::TraitDef> {
        db.lookup_intern_trait(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeAliasId(salsa::InternId);
impl_intern_key!(TypeAliasId);
impl AstItemDef<ast::TypeAliasDef> for TypeAliasId {
    fn intern(db: &impl InternDatabase, loc: ItemLoc<ast::TypeAliasDef>) -> Self {
        db.intern_type_alias(loc)
    }
    fn lookup_intern(self, db: &impl InternDatabase) -> ItemLoc<ast::TypeAliasDef> {
        db.lookup_intern_type_alias(self)
    }
}

macro_rules! impl_froms {
    ($e:ident: $($v:ident $(($($sv:ident),*))?),*) => {
        $(
            impl From<$v> for $e {
                fn from(it: $v) -> $e {
                    $e::$v(it)
                }
            }
            $($(
                impl From<$sv> for $e {
                    fn from(it: $sv) -> $e {
                        $e::$v($v::$sv(it))
                    }
                }
            )*)?
        )*
    }
}

/// A Data Type
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AdtId {
    StructId(StructId),
    UnionId(UnionId),
    EnumId(EnumId),
}
impl_froms!(AdtId: StructId, UnionId, EnumId);

/// The defs which can be visible in the module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModuleDefId {
    ModuleId(ModuleId),
    FunctionId(FunctionId),
    AdtId(AdtId),
    // Can't be directly declared, but can be imported.
    EnumVariantId(EnumVariantId),
    ConstId(ConstId),
    StaticId(StaticId),
    TraitId(TraitId),
    TypeAliasId(TypeAliasId),
    BuiltinType(BuiltinType),
}
impl_froms!(
    ModuleDefId: ModuleId,
    FunctionId,
    AdtId(StructId, EnumId, UnionId),
    EnumVariantId,
    ConstId,
    StaticId,
    TraitId,
    TypeAliasId,
    BuiltinType
);
