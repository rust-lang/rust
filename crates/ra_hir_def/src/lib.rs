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
pub mod diagnostics;
pub mod per_ns;

pub mod adt;
pub mod data;
pub mod generics;
pub mod lang_item;
pub mod docs;

pub mod expr;
pub mod body;
pub mod resolver;

mod trace;
pub mod nameres;

pub mod src;

#[cfg(test)]
mod test_db;
#[cfg(test)]
mod marks;

use std::hash::{Hash, Hasher};

use hir_expand::{ast_id_map::FileAstId, db::AstDatabase, AstId, HirFileId, InFile, MacroDefId};
use ra_arena::{impl_arena_id, RawId};
use ra_db::{impl_intern_key, salsa, CrateId};
use ra_syntax::{ast, AstNode};

use crate::{builtin_type::BuiltinType, db::InternDatabase};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LocalImportId(RawId);
impl_arena_id!(LocalImportId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModuleId {
    pub krate: CrateId,
    pub local_id: LocalModuleId,
}

/// An ID of a module, **local** to a specific crate
// FIXME: rename to `LocalModuleId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LocalModuleId(RawId);
impl_arena_id!(LocalModuleId);

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

pub trait AstItemDef<N: AstNode>: salsa::InternKey + Clone {
    fn intern(db: &impl InternDatabase, loc: ItemLoc<N>) -> Self;
    fn lookup_intern(self, db: &impl InternDatabase) -> ItemLoc<N>;

    fn from_ast_id(ctx: LocationCtx<&impl InternDatabase>, ast_id: FileAstId<N>) -> Self {
        let loc = ItemLoc { module: ctx.module, ast_id: AstId::new(ctx.file_id, ast_id) };
        Self::intern(ctx.db, loc)
    }
    fn source(self, db: &(impl AstDatabase + InternDatabase)) -> InFile<N> {
        let loc = self.lookup_intern(db);
        let value = loc.ast_id.to_node(db);
        InFile { file_id: loc.ast_id.file_id, value }
    }
    fn module(self, db: &impl InternDatabase) -> ModuleId {
        let loc = self.lookup_intern(db);
        loc.module
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FunctionId(salsa::InternId);
impl_intern_key!(FunctionId);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionLoc {
    pub container: ContainerId,
    pub ast_id: AstId<ast::FnDef>,
}

impl Intern for FunctionLoc {
    type ID = FunctionId;
    fn intern(self, db: &impl db::DefDatabase) -> FunctionId {
        db.intern_function(self)
    }
}

impl Lookup for FunctionId {
    type Data = FunctionLoc;
    fn lookup(&self, db: &impl db::DefDatabase) -> FunctionLoc {
        db.lookup_intern_function(*self)
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
impl AstItemDef<ast::UnionDef> for UnionId {
    fn intern(db: &impl InternDatabase, loc: ItemLoc<ast::UnionDef>) -> Self {
        db.intern_union(loc)
    }
    fn lookup_intern(self, db: &impl InternDatabase) -> ItemLoc<ast::UnionDef> {
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
pub struct StructFieldId {
    pub parent: VariantId,
    pub local_id: LocalStructFieldId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LocalStructFieldId(RawId);
impl_arena_id!(LocalStructFieldId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConstId(salsa::InternId);
impl_intern_key!(ConstId);
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConstLoc {
    pub container: ContainerId,
    pub ast_id: AstId<ast::ConstDef>,
}

impl Intern for ConstLoc {
    type ID = ConstId;
    fn intern(self, db: &impl db::DefDatabase) -> ConstId {
        db.intern_const(self)
    }
}

impl Lookup for ConstId {
    type Data = ConstLoc;
    fn lookup(&self, db: &impl db::DefDatabase) -> ConstLoc {
        db.lookup_intern_const(*self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StaticId(salsa::InternId);
impl_intern_key!(StaticId);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StaticLoc {
    pub container: ModuleId,
    pub ast_id: AstId<ast::StaticDef>,
}

impl Intern for StaticLoc {
    type ID = StaticId;
    fn intern(self, db: &impl db::DefDatabase) -> StaticId {
        db.intern_static(self)
    }
}

impl Lookup for StaticId {
    type Data = StaticLoc;
    fn lookup(&self, db: &impl db::DefDatabase) -> StaticLoc {
        db.lookup_intern_static(*self)
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypeAliasLoc {
    pub container: ContainerId,
    pub ast_id: AstId<ast::TypeAliasDef>,
}

impl Intern for TypeAliasLoc {
    type ID = TypeAliasId;
    fn intern(self, db: &impl db::DefDatabase) -> TypeAliasId {
        db.intern_type_alias(self)
    }
}

impl Lookup for TypeAliasId {
    type Data = TypeAliasLoc;
    fn lookup(&self, db: &impl db::DefDatabase) -> TypeAliasLoc {
        db.lookup_intern_type_alias(*self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImplId(salsa::InternId);
impl_intern_key!(ImplId);
impl AstItemDef<ast::ImplBlock> for ImplId {
    fn intern(db: &impl InternDatabase, loc: ItemLoc<ast::ImplBlock>) -> Self {
        db.intern_impl(loc)
    }
    fn lookup_intern(self, db: &impl InternDatabase) -> ItemLoc<ast::ImplBlock> {
        db.lookup_intern_impl(self)
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContainerId {
    ModuleId(ModuleId),
    ImplId(ImplId),
    TraitId(TraitId),
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

/// The defs which have a body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DefWithBodyId {
    FunctionId(FunctionId),
    StaticId(StaticId),
    ConstId(ConstId),
}

impl_froms!(DefWithBodyId: FunctionId, ConstId, StaticId);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum AssocItemId {
    FunctionId(FunctionId),
    ConstId(ConstId),
    TypeAliasId(TypeAliasId),
}
// FIXME: not every function, ... is actually an assoc item. maybe we should make
// sure that you can only turn actual assoc items into AssocItemIds. This would
// require not implementing From, and instead having some checked way of
// casting them, and somehow making the constructors private, which would be annoying.
impl_froms!(AssocItemId: FunctionId, ConstId, TypeAliasId);

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum GenericDefId {
    FunctionId(FunctionId),
    AdtId(AdtId),
    TraitId(TraitId),
    TypeAliasId(TypeAliasId),
    ImplId(ImplId),
    // enum variants cannot have generics themselves, but their parent enums
    // can, and this makes some code easier to write
    EnumVariantId(EnumVariantId),
    // consts can have type parameters from their parents (i.e. associated consts of traits)
    ConstId(ConstId),
}
impl_froms!(
    GenericDefId: FunctionId,
    AdtId(StructId, EnumId, UnionId),
    TraitId,
    TypeAliasId,
    ImplId,
    EnumVariantId,
    ConstId
);

impl From<AssocItemId> for GenericDefId {
    fn from(item: AssocItemId) -> Self {
        match item {
            AssocItemId::FunctionId(f) => f.into(),
            AssocItemId::ConstId(c) => c.into(),
            AssocItemId::TypeAliasId(t) => t.into(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AttrDefId {
    ModuleId(ModuleId),
    StructFieldId(StructFieldId),
    AdtId(AdtId),
    FunctionId(FunctionId),
    EnumVariantId(EnumVariantId),
    StaticId(StaticId),
    ConstId(ConstId),
    TraitId(TraitId),
    TypeAliasId(TypeAliasId),
    MacroDefId(MacroDefId),
    ImplId(ImplId),
}

impl_froms!(
    AttrDefId: ModuleId,
    StructFieldId,
    AdtId(StructId, EnumId, UnionId),
    EnumVariantId,
    StaticId,
    ConstId,
    FunctionId,
    TraitId,
    TypeAliasId,
    MacroDefId,
    ImplId
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VariantId {
    EnumVariantId(EnumVariantId),
    StructId(StructId),
    UnionId(UnionId),
}
impl_froms!(VariantId: EnumVariantId, StructId);

trait Intern {
    type ID;
    fn intern(self, db: &impl db::DefDatabase) -> Self::ID;
}

pub trait Lookup {
    type Data;
    fn lookup(&self, db: &impl db::DefDatabase) -> Self::Data;
}

pub trait HasModule {
    fn module(&self, db: &impl db::DefDatabase) -> ModuleId;
}

impl HasModule for FunctionLoc {
    fn module(&self, db: &impl db::DefDatabase) -> ModuleId {
        match self.container {
            ContainerId::ModuleId(it) => it,
            ContainerId::ImplId(it) => it.module(db),
            ContainerId::TraitId(it) => it.module(db),
        }
    }
}

impl HasModule for TypeAliasLoc {
    fn module(&self, db: &impl db::DefDatabase) -> ModuleId {
        match self.container {
            ContainerId::ModuleId(it) => it,
            ContainerId::ImplId(it) => it.module(db),
            ContainerId::TraitId(it) => it.module(db),
        }
    }
}

impl HasModule for ConstLoc {
    fn module(&self, db: &impl db::DefDatabase) -> ModuleId {
        match self.container {
            ContainerId::ModuleId(it) => it,
            ContainerId::ImplId(it) => it.module(db),
            ContainerId::TraitId(it) => it.module(db),
        }
    }
}

impl HasModule for AdtId {
    fn module(&self, db: &impl db::DefDatabase) -> ModuleId {
        match self {
            AdtId::StructId(it) => it.module(db),
            AdtId::UnionId(it) => it.module(db),
            AdtId::EnumId(it) => it.module(db),
        }
    }
}

impl HasModule for DefWithBodyId {
    fn module(&self, db: &impl db::DefDatabase) -> ModuleId {
        match self {
            DefWithBodyId::FunctionId(it) => it.lookup(db).module(db),
            DefWithBodyId::StaticId(it) => it.lookup(db).module(db),
            DefWithBodyId::ConstId(it) => it.lookup(db).module(db),
        }
    }
}

impl HasModule for StaticLoc {
    fn module(&self, _db: &impl db::DefDatabase) -> ModuleId {
        self.container
    }
}
