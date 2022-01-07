//! `hir_def` crate contains everything between macro expansion and type
//! inference.
//!
//! It defines various items (structs, enums, traits) which comprises Rust code,
//! as well as an algorithm for resolving paths to such entities.
//!
//! Note that `hir_def` is a work in progress, so not all of the above is
//! actually true.

#[allow(unused)]
macro_rules! eprintln {
    ($($tt:tt)*) => { stdx::eprintln!($($tt)*) };
}

pub mod db;

pub mod attr;
pub mod path;
pub mod type_ref;
pub mod builtin_type;
pub mod builtin_attr;
pub mod per_ns;
pub mod item_scope;

pub mod dyn_map;
pub mod keys;

pub mod item_tree;
pub mod intern;

pub mod adt;
pub mod data;
pub mod generics;
pub mod lang_item;

pub mod expr;
pub mod body;
pub mod resolver;

mod trace;
pub mod nameres;

pub mod src;
pub mod child_by_source;

pub mod visibility;
pub mod find_path;
pub mod import_map;

#[cfg(test)]
mod test_db;
#[cfg(test)]
mod macro_expansion_tests;

use std::{
    hash::{Hash, Hasher},
    sync::Arc,
};

use attr::Attr;
use base_db::{impl_intern_key, salsa, CrateId};
use hir_expand::{
    ast_id_map::FileAstId,
    eager::{expand_eager_macro, ErrorEmitted, ErrorSink},
    hygiene::Hygiene,
    AstId, ExpandTo, HirFileId, InFile, MacroCallId, MacroCallKind, MacroDefId, MacroDefKind,
};
use item_tree::ExternBlock;
use la_arena::Idx;
use nameres::DefMap;
use path::ModPath;
use stdx::impl_from;
use syntax::ast;

use crate::{
    adt::VariantData,
    attr::AttrId,
    builtin_type::BuiltinType,
    item_tree::{
        Const, Enum, Function, Impl, ItemTreeId, ItemTreeNode, ModItem, Static, Struct, Trait,
        TypeAlias, Union,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModuleId {
    krate: CrateId,
    /// If this `ModuleId` was derived from a `DefMap` for a block expression, this stores the
    /// `BlockId` of that block expression. If `None`, this module is part of the crate-level
    /// `DefMap` of `krate`.
    block: Option<BlockId>,
    /// The module's ID in its originating `DefMap`.
    pub local_id: LocalModuleId,
}

impl ModuleId {
    pub fn def_map(&self, db: &dyn db::DefDatabase) -> Arc<DefMap> {
        match self.block {
            Some(block) => {
                db.block_def_map(block).unwrap_or_else(|| {
                    // NOTE: This should be unreachable - all `ModuleId`s come from their `DefMap`s,
                    // so the `DefMap` here must exist.
                    unreachable!("no `block_def_map` for `ModuleId` {:?}", self);
                })
            }
            None => db.crate_def_map(self.krate),
        }
    }

    pub fn krate(&self) -> CrateId {
        self.krate
    }

    pub fn containing_module(&self, db: &dyn db::DefDatabase) -> Option<ModuleId> {
        self.def_map(db).containing_module(self.local_id)
    }

    pub fn containing_block(&self) -> Option<BlockId> {
        self.block
    }
}

/// An ID of a module, **local** to a specific crate
pub type LocalModuleId = Idx<nameres::ModuleData>;

#[derive(Debug)]
pub struct ItemLoc<N: ItemTreeNode> {
    pub container: ModuleId,
    pub id: ItemTreeId<N>,
}

impl<N: ItemTreeNode> Clone for ItemLoc<N> {
    fn clone(&self) -> Self {
        Self { container: self.container, id: self.id }
    }
}

impl<N: ItemTreeNode> Copy for ItemLoc<N> {}

impl<N: ItemTreeNode> PartialEq for ItemLoc<N> {
    fn eq(&self, other: &Self) -> bool {
        self.container == other.container && self.id == other.id
    }
}

impl<N: ItemTreeNode> Eq for ItemLoc<N> {}

impl<N: ItemTreeNode> Hash for ItemLoc<N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.container.hash(state);
        self.id.hash(state);
    }
}

#[derive(Debug)]
pub struct AssocItemLoc<N: ItemTreeNode> {
    pub container: ItemContainerId,
    pub id: ItemTreeId<N>,
}

impl<N: ItemTreeNode> Clone for AssocItemLoc<N> {
    fn clone(&self) -> Self {
        Self { container: self.container, id: self.id }
    }
}

impl<N: ItemTreeNode> Copy for AssocItemLoc<N> {}

impl<N: ItemTreeNode> PartialEq for AssocItemLoc<N> {
    fn eq(&self, other: &Self) -> bool {
        self.container == other.container && self.id == other.id
    }
}

impl<N: ItemTreeNode> Eq for AssocItemLoc<N> {}

impl<N: ItemTreeNode> Hash for AssocItemLoc<N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.container.hash(state);
        self.id.hash(state);
    }
}

macro_rules! impl_intern {
    ($id:ident, $loc:ident, $intern:ident, $lookup:ident) => {
        impl_intern_key!($id);

        impl Intern for $loc {
            type ID = $id;
            fn intern(self, db: &dyn db::DefDatabase) -> $id {
                db.$intern(self)
            }
        }

        impl Lookup for $id {
            type Data = $loc;
            fn lookup(&self, db: &dyn db::DefDatabase) -> $loc {
                db.$lookup(*self)
            }
        }
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FunctionId(salsa::InternId);
type FunctionLoc = AssocItemLoc<Function>;
impl_intern!(FunctionId, FunctionLoc, intern_function, lookup_intern_function);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StructId(salsa::InternId);
type StructLoc = ItemLoc<Struct>;
impl_intern!(StructId, StructLoc, intern_struct, lookup_intern_struct);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct UnionId(salsa::InternId);
pub type UnionLoc = ItemLoc<Union>;
impl_intern!(UnionId, UnionLoc, intern_union, lookup_intern_union);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EnumId(salsa::InternId);
pub type EnumLoc = ItemLoc<Enum>;
impl_intern!(EnumId, EnumLoc, intern_enum, lookup_intern_enum);

// FIXME: rename to `VariantId`, only enums can ave variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EnumVariantId {
    pub parent: EnumId,
    pub local_id: LocalEnumVariantId,
}

pub type LocalEnumVariantId = Idx<adt::EnumVariantData>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FieldId {
    pub parent: VariantId,
    pub local_id: LocalFieldId,
}

pub type LocalFieldId = Idx<adt::FieldData>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConstId(salsa::InternId);
type ConstLoc = AssocItemLoc<Const>;
impl_intern!(ConstId, ConstLoc, intern_const, lookup_intern_const);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StaticId(salsa::InternId);
pub type StaticLoc = AssocItemLoc<Static>;
impl_intern!(StaticId, StaticLoc, intern_static, lookup_intern_static);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TraitId(salsa::InternId);
pub type TraitLoc = ItemLoc<Trait>;
impl_intern!(TraitId, TraitLoc, intern_trait, lookup_intern_trait);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeAliasId(salsa::InternId);
type TypeAliasLoc = AssocItemLoc<TypeAlias>;
impl_intern!(TypeAliasId, TypeAliasLoc, intern_type_alias, lookup_intern_type_alias);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct ImplId(salsa::InternId);
type ImplLoc = ItemLoc<Impl>;
impl_intern!(ImplId, ImplLoc, intern_impl, lookup_intern_impl);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct ExternBlockId(salsa::InternId);
type ExternBlockLoc = ItemLoc<ExternBlock>;
impl_intern!(ExternBlockId, ExternBlockLoc, intern_extern_block, lookup_intern_extern_block);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct BlockId(salsa::InternId);
#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub struct BlockLoc {
    ast_id: AstId<ast::BlockExpr>,
    /// The containing module.
    module: ModuleId,
}
impl_intern!(BlockId, BlockLoc, intern_block, lookup_intern_block);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeParamId {
    pub parent: GenericDefId,
    pub local_id: LocalTypeParamId,
}

pub type LocalTypeParamId = Idx<generics::TypeParamData>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LifetimeParamId {
    pub parent: GenericDefId,
    pub local_id: LocalLifetimeParamId,
}
pub type LocalLifetimeParamId = Idx<generics::LifetimeParamData>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConstParamId {
    pub parent: GenericDefId,
    pub local_id: LocalConstParamId,
}
pub type LocalConstParamId = Idx<generics::ConstParamData>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ItemContainerId {
    ExternBlockId(ExternBlockId),
    ModuleId(ModuleId),
    ImplId(ImplId),
    TraitId(TraitId),
}
impl_from!(ModuleId for ItemContainerId);

/// A Data Type
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AdtId {
    StructId(StructId),
    UnionId(UnionId),
    EnumId(EnumId),
}
impl_from!(StructId, UnionId, EnumId for AdtId);

/// A generic param
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GenericParamId {
    TypeParamId(TypeParamId),
    LifetimeParamId(LifetimeParamId),
    ConstParamId(ConstParamId),
}
impl_from!(TypeParamId, LifetimeParamId, ConstParamId for GenericParamId);

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
impl_from!(
    ModuleId,
    FunctionId,
    AdtId(StructId, EnumId, UnionId),
    EnumVariantId,
    ConstId,
    StaticId,
    TraitId,
    TypeAliasId,
    BuiltinType
    for ModuleDefId
);

/// The defs which have a body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DefWithBodyId {
    FunctionId(FunctionId),
    StaticId(StaticId),
    ConstId(ConstId),
}

impl_from!(FunctionId, ConstId, StaticId for DefWithBodyId);

impl DefWithBodyId {
    pub fn as_generic_def_id(self) -> Option<GenericDefId> {
        match self {
            DefWithBodyId::FunctionId(f) => Some(f.into()),
            DefWithBodyId::StaticId(_) => None,
            DefWithBodyId::ConstId(c) => Some(c.into()),
        }
    }
}

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
impl_from!(FunctionId, ConstId, TypeAliasId for AssocItemId);

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
impl_from!(
    FunctionId,
    AdtId(StructId, EnumId, UnionId),
    TraitId,
    TypeAliasId,
    ImplId,
    EnumVariantId,
    ConstId
    for GenericDefId
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
    FieldId(FieldId),
    AdtId(AdtId),
    FunctionId(FunctionId),
    EnumVariantId(EnumVariantId),
    StaticId(StaticId),
    ConstId(ConstId),
    TraitId(TraitId),
    TypeAliasId(TypeAliasId),
    MacroDefId(MacroDefId),
    ImplId(ImplId),
    GenericParamId(GenericParamId),
    ExternBlockId(ExternBlockId),
}

impl_from!(
    ModuleId,
    FieldId,
    AdtId(StructId, EnumId, UnionId),
    EnumVariantId,
    StaticId,
    ConstId,
    FunctionId,
    TraitId,
    TypeAliasId,
    MacroDefId,
    ImplId,
    GenericParamId
    for AttrDefId
);

impl From<ItemContainerId> for AttrDefId {
    fn from(acid: ItemContainerId) -> Self {
        match acid {
            ItemContainerId::ModuleId(mid) => AttrDefId::ModuleId(mid),
            ItemContainerId::ImplId(iid) => AttrDefId::ImplId(iid),
            ItemContainerId::TraitId(tid) => AttrDefId::TraitId(tid),
            ItemContainerId::ExternBlockId(id) => AttrDefId::ExternBlockId(id),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VariantId {
    EnumVariantId(EnumVariantId),
    StructId(StructId),
    UnionId(UnionId),
}
impl_from!(EnumVariantId, StructId, UnionId for VariantId);

impl VariantId {
    pub fn variant_data(self, db: &dyn db::DefDatabase) -> Arc<VariantData> {
        match self {
            VariantId::StructId(it) => db.struct_data(it).variant_data.clone(),
            VariantId::UnionId(it) => db.union_data(it).variant_data.clone(),
            VariantId::EnumVariantId(it) => {
                db.enum_data(it.parent).variants[it.local_id].variant_data.clone()
            }
        }
    }

    pub fn file_id(self, db: &dyn db::DefDatabase) -> HirFileId {
        match self {
            VariantId::EnumVariantId(it) => it.parent.lookup(db).id.file_id(),
            VariantId::StructId(it) => it.lookup(db).id.file_id(),
            VariantId::UnionId(it) => it.lookup(db).id.file_id(),
        }
    }

    pub fn adt_id(self) -> AdtId {
        match self {
            VariantId::EnumVariantId(it) => it.parent.into(),
            VariantId::StructId(it) => it.into(),
            VariantId::UnionId(it) => it.into(),
        }
    }
}

trait Intern {
    type ID;
    fn intern(self, db: &dyn db::DefDatabase) -> Self::ID;
}

pub trait Lookup {
    type Data;
    fn lookup(&self, db: &dyn db::DefDatabase) -> Self::Data;
}

pub trait HasModule {
    fn module(&self, db: &dyn db::DefDatabase) -> ModuleId;
}

impl HasModule for ItemContainerId {
    fn module(&self, db: &dyn db::DefDatabase) -> ModuleId {
        match *self {
            ItemContainerId::ModuleId(it) => it,
            ItemContainerId::ImplId(it) => it.lookup(db).container,
            ItemContainerId::TraitId(it) => it.lookup(db).container,
            ItemContainerId::ExternBlockId(it) => it.lookup(db).container,
        }
    }
}

impl<N: ItemTreeNode> HasModule for AssocItemLoc<N> {
    fn module(&self, db: &dyn db::DefDatabase) -> ModuleId {
        self.container.module(db)
    }
}

impl HasModule for AdtId {
    fn module(&self, db: &dyn db::DefDatabase) -> ModuleId {
        match self {
            AdtId::StructId(it) => it.lookup(db).container,
            AdtId::UnionId(it) => it.lookup(db).container,
            AdtId::EnumId(it) => it.lookup(db).container,
        }
    }
}

impl HasModule for VariantId {
    fn module(&self, db: &dyn db::DefDatabase) -> ModuleId {
        match self {
            VariantId::EnumVariantId(it) => it.parent.lookup(db).container,
            VariantId::StructId(it) => it.lookup(db).container,
            VariantId::UnionId(it) => it.lookup(db).container,
        }
    }
}

impl HasModule for DefWithBodyId {
    fn module(&self, db: &dyn db::DefDatabase) -> ModuleId {
        match self {
            DefWithBodyId::FunctionId(it) => it.lookup(db).module(db),
            DefWithBodyId::StaticId(it) => it.lookup(db).module(db),
            DefWithBodyId::ConstId(it) => it.lookup(db).module(db),
        }
    }
}

impl DefWithBodyId {
    pub fn as_mod_item(self, db: &dyn db::DefDatabase) -> ModItem {
        match self {
            DefWithBodyId::FunctionId(it) => it.lookup(db).id.value.into(),
            DefWithBodyId::StaticId(it) => it.lookup(db).id.value.into(),
            DefWithBodyId::ConstId(it) => it.lookup(db).id.value.into(),
        }
    }
}

impl HasModule for GenericDefId {
    fn module(&self, db: &dyn db::DefDatabase) -> ModuleId {
        match self {
            GenericDefId::FunctionId(it) => it.lookup(db).module(db),
            GenericDefId::AdtId(it) => it.module(db),
            GenericDefId::TraitId(it) => it.lookup(db).container,
            GenericDefId::TypeAliasId(it) => it.lookup(db).module(db),
            GenericDefId::ImplId(it) => it.lookup(db).container,
            GenericDefId::EnumVariantId(it) => it.parent.lookup(db).container,
            GenericDefId::ConstId(it) => it.lookup(db).module(db),
        }
    }
}

impl HasModule for TypeAliasId {
    fn module(&self, db: &dyn db::DefDatabase) -> ModuleId {
        self.lookup(db).module(db)
    }
}

impl HasModule for TraitId {
    fn module(&self, db: &dyn db::DefDatabase) -> ModuleId {
        self.lookup(db).container
    }
}

impl ModuleDefId {
    /// Returns the module containing `self` (or `self`, if `self` is itself a module).
    ///
    /// Returns `None` if `self` refers to a primitive type.
    pub fn module(&self, db: &dyn db::DefDatabase) -> Option<ModuleId> {
        Some(match self {
            ModuleDefId::ModuleId(id) => *id,
            ModuleDefId::FunctionId(id) => id.lookup(db).module(db),
            ModuleDefId::AdtId(id) => id.module(db),
            ModuleDefId::EnumVariantId(id) => id.parent.lookup(db).container,
            ModuleDefId::ConstId(id) => id.lookup(db).container.module(db),
            ModuleDefId::StaticId(id) => id.lookup(db).module(db),
            ModuleDefId::TraitId(id) => id.lookup(db).container,
            ModuleDefId::TypeAliasId(id) => id.lookup(db).module(db),
            ModuleDefId::BuiltinType(_) => return None,
        })
    }
}

impl AttrDefId {
    pub fn krate(&self, db: &dyn db::DefDatabase) -> CrateId {
        match self {
            AttrDefId::ModuleId(it) => it.krate,
            AttrDefId::FieldId(it) => it.parent.module(db).krate,
            AttrDefId::AdtId(it) => it.module(db).krate,
            AttrDefId::FunctionId(it) => it.lookup(db).module(db).krate,
            AttrDefId::EnumVariantId(it) => it.parent.lookup(db).container.krate,
            AttrDefId::StaticId(it) => it.lookup(db).module(db).krate,
            AttrDefId::ConstId(it) => it.lookup(db).module(db).krate,
            AttrDefId::TraitId(it) => it.lookup(db).container.krate,
            AttrDefId::TypeAliasId(it) => it.lookup(db).module(db).krate,
            AttrDefId::ImplId(it) => it.lookup(db).container.krate,
            AttrDefId::ExternBlockId(it) => it.lookup(db).container.krate,
            AttrDefId::GenericParamId(it) => {
                match it {
                    GenericParamId::TypeParamId(it) => it.parent,
                    GenericParamId::LifetimeParamId(it) => it.parent,
                    GenericParamId::ConstParamId(it) => it.parent,
                }
                .module(db)
                .krate
            }
            // FIXME: `MacroDefId` should store the defining module, then this can implement
            // `HasModule`
            AttrDefId::MacroDefId(it) => it.krate,
        }
    }
}

/// A helper trait for converting to MacroCallId
pub trait AsMacroCall {
    fn as_call_id(
        &self,
        db: &dyn db::DefDatabase,
        krate: CrateId,
        resolver: impl Fn(path::ModPath) -> Option<MacroDefId>,
    ) -> Option<MacroCallId> {
        self.as_call_id_with_errors(db, krate, resolver, &mut |_| ()).ok()?.ok()
    }

    fn as_call_id_with_errors(
        &self,
        db: &dyn db::DefDatabase,
        krate: CrateId,
        resolver: impl Fn(path::ModPath) -> Option<MacroDefId>,
        error_sink: &mut dyn FnMut(mbe::ExpandError),
    ) -> Result<Result<MacroCallId, ErrorEmitted>, UnresolvedMacro>;
}

impl AsMacroCall for InFile<&ast::MacroCall> {
    fn as_call_id_with_errors(
        &self,
        db: &dyn db::DefDatabase,
        krate: CrateId,
        resolver: impl Fn(path::ModPath) -> Option<MacroDefId>,
        mut error_sink: &mut dyn FnMut(mbe::ExpandError),
    ) -> Result<Result<MacroCallId, ErrorEmitted>, UnresolvedMacro> {
        let expands_to = hir_expand::ExpandTo::from_call_site(self.value);
        let ast_id = AstId::new(self.file_id, db.ast_id_map(self.file_id).ast_id(self.value));
        let h = Hygiene::new(db.upcast(), self.file_id);
        let path = self.value.path().and_then(|path| path::ModPath::from_src(db, path, &h));

        let path = match error_sink
            .option(path, || mbe::ExpandError::Other("malformed macro invocation".into()))
        {
            Ok(path) => path,
            Err(error) => {
                return Ok(Err(error));
            }
        };

        macro_call_as_call_id(
            &AstIdWithPath::new(ast_id.file_id, ast_id.value, path),
            expands_to,
            db,
            krate,
            resolver,
            error_sink,
        )
    }
}

/// Helper wrapper for `AstId` with `ModPath`
#[derive(Clone, Debug, Eq, PartialEq)]
struct AstIdWithPath<T: ast::AstNode> {
    ast_id: AstId<T>,
    path: path::ModPath,
}

impl<T: ast::AstNode> AstIdWithPath<T> {
    fn new(file_id: HirFileId, ast_id: FileAstId<T>, path: path::ModPath) -> AstIdWithPath<T> {
        AstIdWithPath { ast_id: AstId::new(file_id, ast_id), path }
    }
}

#[derive(Debug)]
pub struct UnresolvedMacro {
    pub path: ModPath,
}

fn macro_call_as_call_id(
    call: &AstIdWithPath<ast::MacroCall>,
    expand_to: ExpandTo,
    db: &dyn db::DefDatabase,
    krate: CrateId,
    resolver: impl Fn(path::ModPath) -> Option<MacroDefId>,
    error_sink: &mut dyn FnMut(mbe::ExpandError),
) -> Result<Result<MacroCallId, ErrorEmitted>, UnresolvedMacro> {
    let def: MacroDefId =
        resolver(call.path.clone()).ok_or_else(|| UnresolvedMacro { path: call.path.clone() })?;

    let res = if let MacroDefKind::BuiltInEager(..) = def.kind {
        let macro_call = InFile::new(call.ast_id.file_id, call.ast_id.to_node(db.upcast()));
        let hygiene = Hygiene::new(db.upcast(), call.ast_id.file_id);

        expand_eager_macro(
            db.upcast(),
            krate,
            macro_call,
            def,
            &|path: ast::Path| resolver(path::ModPath::from_src(db, path, &hygiene)?),
            error_sink,
        )
    } else {
        Ok(def.as_lazy_macro(
            db.upcast(),
            krate,
            MacroCallKind::FnLike { ast_id: call.ast_id, expand_to },
        ))
    };
    Ok(res)
}

fn derive_macro_as_call_id(
    item_attr: &AstIdWithPath<ast::Adt>,
    derive_attr: AttrId,
    db: &dyn db::DefDatabase,
    krate: CrateId,
    resolver: impl Fn(path::ModPath) -> Option<MacroDefId>,
) -> Result<MacroCallId, UnresolvedMacro> {
    let def: MacroDefId = resolver(item_attr.path.clone())
        .ok_or_else(|| UnresolvedMacro { path: item_attr.path.clone() })?;
    let last_segment = item_attr
        .path
        .segments()
        .last()
        .ok_or_else(|| UnresolvedMacro { path: item_attr.path.clone() })?;
    let res = def.as_lazy_macro(
        db.upcast(),
        krate,
        MacroCallKind::Derive {
            ast_id: item_attr.ast_id,
            derive_name: last_segment.to_string().into_boxed_str(),
            derive_attr_index: derive_attr.ast_index,
        },
    );
    Ok(res)
}

fn attr_macro_as_call_id(
    item_attr: &AstIdWithPath<ast::Item>,
    macro_attr: &Attr,
    db: &dyn db::DefDatabase,
    krate: CrateId,
    def: MacroDefId,
) -> MacroCallId {
    let attr_path = &item_attr.path;
    let last_segment = attr_path.segments().last().expect("empty attribute path");
    let mut arg = match macro_attr.input.as_deref() {
        Some(attr::AttrInput::TokenTree(tt, map)) => (tt.clone(), map.clone()),
        _ => Default::default(),
    };

    // The parentheses are always disposed here.
    arg.0.delimiter = None;

    let res = def.as_lazy_macro(
        db.upcast(),
        krate,
        MacroCallKind::Attr {
            ast_id: item_attr.ast_id,
            attr_name: last_segment.to_string().into_boxed_str(),
            attr_args: arg,
            invoc_attr_index: macro_attr.id.ast_index,
        },
    );
    res
}
