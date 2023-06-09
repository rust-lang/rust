//! `hir_def` crate contains everything between macro expansion and type
//! inference.
//!
//! It defines various items (structs, enums, traits) which comprises Rust code,
//! as well as an algorithm for resolving paths to such entities.
//!
//! Note that `hir_def` is a work in progress, so not all of the above is
//! actually true.

#![warn(rust_2018_idioms, unused_lifetimes, semicolon_in_expressions_from_macros)]

#[allow(unused)]
macro_rules! eprintln {
    ($($tt:tt)*) => { stdx::eprintln!($($tt)*) };
}

pub mod db;

pub mod attr;
pub mod path;
pub mod builtin_type;
pub mod per_ns;
pub mod item_scope;

pub mod lower;
pub mod expander;

pub mod dyn_map;

pub mod item_tree;

pub mod data;
pub mod generics;
pub mod lang_item;

pub mod hir;
pub use self::hir::type_ref;
pub mod body;
pub mod resolver;

mod trace;
pub mod nameres;

pub mod src;
pub mod child_by_source;

pub mod visibility;
pub mod find_path;
pub mod import_map;

pub use rustc_abi as layout;
use triomphe::Arc;

#[cfg(test)]
mod test_db;
#[cfg(test)]
mod macro_expansion_tests;
mod pretty;

use std::hash::{Hash, Hasher};

use base_db::{
    impl_intern_key,
    salsa::{self, InternId},
    CrateId, ProcMacroKind,
};
use hir_expand::{
    ast_id_map::FileAstId,
    attrs::{Attr, AttrId, AttrInput},
    builtin_attr_macro::BuiltinAttrExpander,
    builtin_derive_macro::BuiltinDeriveExpander,
    builtin_fn_macro::{BuiltinFnLikeExpander, EagerExpander},
    db::ExpandDatabase,
    eager::expand_eager_macro,
    hygiene::Hygiene,
    proc_macro::ProcMacroExpander,
    AstId, ExpandError, ExpandResult, ExpandTo, HirFileId, InFile, MacroCallId, MacroCallKind,
    MacroDefId, MacroDefKind, UnresolvedMacro,
};
use item_tree::ExternBlock;
use la_arena::Idx;
use nameres::DefMap;
use stdx::impl_from;
use syntax::ast;

use ::tt::token_id as tt;

use crate::{
    builtin_type::BuiltinType,
    data::adt::VariantData,
    item_tree::{
        Const, Enum, Function, Impl, ItemTreeId, ItemTreeNode, MacroDef, MacroRules, ModItem,
        Static, Struct, Trait, TraitAlias, TypeAlias, Union,
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
            Some(block) => db.block_def_map(block),
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

/// An ID of a module, **local** to a `DefMap`.
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

pub type LocalEnumVariantId = Idx<data::adt::EnumVariantData>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FieldId {
    pub parent: VariantId,
    pub local_id: LocalFieldId,
}

pub type LocalFieldId = Idx<data::adt::FieldData>;

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
pub struct TraitAliasId(salsa::InternId);
pub type TraitAliasLoc = ItemLoc<TraitAlias>;
impl_intern!(TraitAliasId, TraitAliasLoc, intern_trait_alias, lookup_intern_trait_alias);

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MacroExpander {
    Declarative,
    BuiltIn(BuiltinFnLikeExpander),
    BuiltInAttr(BuiltinAttrExpander),
    BuiltInDerive(BuiltinDeriveExpander),
    BuiltInEager(EagerExpander),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct Macro2Id(salsa::InternId);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Macro2Loc {
    pub container: ModuleId,
    pub id: ItemTreeId<MacroDef>,
    pub expander: MacroExpander,
    pub allow_internal_unsafe: bool,
}
impl_intern!(Macro2Id, Macro2Loc, intern_macro2, lookup_intern_macro2);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct MacroRulesId(salsa::InternId);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MacroRulesLoc {
    pub container: ModuleId,
    pub id: ItemTreeId<MacroRules>,
    pub expander: MacroExpander,
    pub allow_internal_unsafe: bool,
    pub local_inner: bool,
}
impl_intern!(MacroRulesId, MacroRulesLoc, intern_macro_rules, lookup_intern_macro_rules);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct ProcMacroId(salsa::InternId);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProcMacroLoc {
    // FIXME: this should be a crate? or just a crate-root module
    pub container: ModuleId,
    pub id: ItemTreeId<Function>,
    pub expander: ProcMacroExpander,
    pub kind: ProcMacroKind,
}
impl_intern!(ProcMacroId, ProcMacroLoc, intern_proc_macro, lookup_intern_proc_macro);

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
pub struct TypeOrConstParamId {
    pub parent: GenericDefId,
    pub local_id: LocalTypeOrConstParamId,
}

/// A TypeOrConstParamId with an invariant that it actually belongs to a type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeParamId(TypeOrConstParamId);

impl TypeParamId {
    pub fn parent(&self) -> GenericDefId {
        self.0.parent
    }
    pub fn local_id(&self) -> LocalTypeOrConstParamId {
        self.0.local_id
    }
}

impl TypeParamId {
    /// Caller should check if this toc id really belongs to a type
    pub fn from_unchecked(x: TypeOrConstParamId) -> Self {
        Self(x)
    }
}

impl From<TypeParamId> for TypeOrConstParamId {
    fn from(x: TypeParamId) -> Self {
        x.0
    }
}

/// A TypeOrConstParamId with an invariant that it actually belongs to a const
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConstParamId(TypeOrConstParamId);

impl ConstParamId {
    pub fn parent(&self) -> GenericDefId {
        self.0.parent
    }
    pub fn local_id(&self) -> LocalTypeOrConstParamId {
        self.0.local_id
    }
}

impl ConstParamId {
    /// Caller should check if this toc id really belongs to a const
    pub fn from_unchecked(x: TypeOrConstParamId) -> Self {
        Self(x)
    }
}

impl From<ConstParamId> for TypeOrConstParamId {
    fn from(x: ConstParamId) -> Self {
        x.0
    }
}

pub type LocalTypeOrConstParamId = Idx<generics::TypeOrConstParamData>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LifetimeParamId {
    pub parent: GenericDefId,
    pub local_id: LocalLifetimeParamId,
}
pub type LocalLifetimeParamId = Idx<generics::LifetimeParamData>;

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

/// A macro
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MacroId {
    Macro2Id(Macro2Id),
    MacroRulesId(MacroRulesId),
    ProcMacroId(ProcMacroId),
}
impl_from!(Macro2Id, MacroRulesId, ProcMacroId for MacroId);

impl MacroId {
    pub fn is_attribute(self, db: &dyn db::DefDatabase) -> bool {
        match self {
            MacroId::ProcMacroId(it) => it.lookup(db).kind == ProcMacroKind::Attr,
            _ => false,
        }
    }
}

/// A generic param
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GenericParamId {
    TypeParamId(TypeParamId),
    ConstParamId(ConstParamId),
    LifetimeParamId(LifetimeParamId),
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
    TraitAliasId(TraitAliasId),
    TypeAliasId(TypeAliasId),
    BuiltinType(BuiltinType),
    MacroId(MacroId),
}
impl_from!(
    MacroId(Macro2Id, MacroRulesId, ProcMacroId),
    ModuleId,
    FunctionId,
    AdtId(StructId, EnumId, UnionId),
    EnumVariantId,
    ConstId,
    StaticId,
    TraitId,
    TraitAliasId,
    TypeAliasId,
    BuiltinType
    for ModuleDefId
);

// FIXME: make this a DefWithBodyId
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct AnonymousConstId(InternId);
impl_intern_key!(AnonymousConstId);

/// A constant, which might appears as a const item, an annonymous const block in expressions
/// or patterns, or as a constant in types with const generics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GeneralConstId {
    ConstId(ConstId),
    AnonymousConstId(AnonymousConstId),
}

impl_from!(ConstId, AnonymousConstId for GeneralConstId);

impl GeneralConstId {
    pub fn generic_def(self, db: &dyn db::DefDatabase) -> Option<GenericDefId> {
        match self {
            GeneralConstId::ConstId(x) => Some(x.into()),
            GeneralConstId::AnonymousConstId(x) => {
                let (parent, _) = db.lookup_intern_anonymous_const(x);
                parent.as_generic_def_id()
            }
        }
    }

    pub fn name(self, db: &dyn db::DefDatabase) -> String {
        match self {
            GeneralConstId::ConstId(const_id) => db
                .const_data(const_id)
                .name
                .as_ref()
                .and_then(|x| x.as_str())
                .unwrap_or("_")
                .to_owned(),
            GeneralConstId::AnonymousConstId(id) => format!("{{anonymous const {id:?}}}"),
        }
    }
}

/// The defs which have a body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DefWithBodyId {
    FunctionId(FunctionId),
    StaticId(StaticId),
    ConstId(ConstId),
    VariantId(EnumVariantId),
}

impl_from!(FunctionId, ConstId, StaticId for DefWithBodyId);

impl From<EnumVariantId> for DefWithBodyId {
    fn from(id: EnumVariantId) -> Self {
        DefWithBodyId::VariantId(id)
    }
}

impl DefWithBodyId {
    pub fn as_generic_def_id(self) -> Option<GenericDefId> {
        match self {
            DefWithBodyId::FunctionId(f) => Some(f.into()),
            DefWithBodyId::StaticId(_) => None,
            DefWithBodyId::ConstId(c) => Some(c.into()),
            DefWithBodyId::VariantId(c) => Some(c.into()),
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
    TraitAliasId(TraitAliasId),
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
    TraitAliasId,
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
    TraitAliasId(TraitAliasId),
    TypeAliasId(TypeAliasId),
    MacroId(MacroId),
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
    MacroId(Macro2Id, MacroRulesId, ProcMacroId),
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

impl HasModule for MacroId {
    fn module(&self, db: &dyn db::DefDatabase) -> ModuleId {
        match self {
            MacroId::MacroRulesId(it) => it.lookup(db).container,
            MacroId::Macro2Id(it) => it.lookup(db).container,
            MacroId::ProcMacroId(it) => it.lookup(db).container,
        }
    }
}

impl HasModule for DefWithBodyId {
    fn module(&self, db: &dyn db::DefDatabase) -> ModuleId {
        match self {
            DefWithBodyId::FunctionId(it) => it.lookup(db).module(db),
            DefWithBodyId::StaticId(it) => it.lookup(db).module(db),
            DefWithBodyId::ConstId(it) => it.lookup(db).module(db),
            DefWithBodyId::VariantId(it) => it.parent.lookup(db).container,
        }
    }
}

impl DefWithBodyId {
    pub fn as_mod_item(self, db: &dyn db::DefDatabase) -> ModItem {
        match self {
            DefWithBodyId::FunctionId(it) => it.lookup(db).id.value.into(),
            DefWithBodyId::StaticId(it) => it.lookup(db).id.value.into(),
            DefWithBodyId::ConstId(it) => it.lookup(db).id.value.into(),
            DefWithBodyId::VariantId(it) => it.parent.lookup(db).id.value.into(),
        }
    }
}

impl HasModule for GenericDefId {
    fn module(&self, db: &dyn db::DefDatabase) -> ModuleId {
        match self {
            GenericDefId::FunctionId(it) => it.lookup(db).module(db),
            GenericDefId::AdtId(it) => it.module(db),
            GenericDefId::TraitId(it) => it.lookup(db).container,
            GenericDefId::TraitAliasId(it) => it.lookup(db).container,
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
            ModuleDefId::TraitAliasId(id) => id.lookup(db).container,
            ModuleDefId::TypeAliasId(id) => id.lookup(db).module(db),
            ModuleDefId::MacroId(id) => id.module(db),
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
            AttrDefId::TraitAliasId(it) => it.lookup(db).container.krate,
            AttrDefId::TypeAliasId(it) => it.lookup(db).module(db).krate,
            AttrDefId::ImplId(it) => it.lookup(db).container.krate,
            AttrDefId::ExternBlockId(it) => it.lookup(db).container.krate,
            AttrDefId::GenericParamId(it) => {
                match it {
                    GenericParamId::TypeParamId(it) => it.parent(),
                    GenericParamId::ConstParamId(it) => it.parent(),
                    GenericParamId::LifetimeParamId(it) => it.parent,
                }
                .module(db)
                .krate
            }
            AttrDefId::MacroId(it) => it.module(db).krate,
        }
    }
}

/// A helper trait for converting to MacroCallId
pub trait AsMacroCall {
    fn as_call_id(
        &self,
        db: &dyn ExpandDatabase,
        krate: CrateId,
        resolver: impl Fn(path::ModPath) -> Option<MacroDefId>,
    ) -> Option<MacroCallId> {
        self.as_call_id_with_errors(db, krate, resolver).ok()?.value
    }

    fn as_call_id_with_errors(
        &self,
        db: &dyn ExpandDatabase,
        krate: CrateId,
        resolver: impl Fn(path::ModPath) -> Option<MacroDefId>,
    ) -> Result<ExpandResult<Option<MacroCallId>>, UnresolvedMacro>;
}

impl AsMacroCall for InFile<&ast::MacroCall> {
    fn as_call_id_with_errors(
        &self,
        db: &dyn ExpandDatabase,
        krate: CrateId,
        resolver: impl Fn(path::ModPath) -> Option<MacroDefId>,
    ) -> Result<ExpandResult<Option<MacroCallId>>, UnresolvedMacro> {
        let expands_to = hir_expand::ExpandTo::from_call_site(self.value);
        let ast_id = AstId::new(self.file_id, db.ast_id_map(self.file_id).ast_id(self.value));
        let h = Hygiene::new(db, self.file_id);
        let path = self.value.path().and_then(|path| path::ModPath::from_src(db, path, &h));

        let Some(path) = path else {
            return Ok(ExpandResult::only_err(ExpandError::Other("malformed macro invocation".into())));
        };

        macro_call_as_call_id_(
            db,
            &AstIdWithPath::new(ast_id.file_id, ast_id.value, path),
            expands_to,
            krate,
            resolver,
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

fn macro_call_as_call_id(
    db: &dyn ExpandDatabase,
    call: &AstIdWithPath<ast::MacroCall>,
    expand_to: ExpandTo,
    krate: CrateId,
    resolver: impl Fn(path::ModPath) -> Option<MacroDefId>,
) -> Result<Option<MacroCallId>, UnresolvedMacro> {
    macro_call_as_call_id_(db, call, expand_to, krate, resolver).map(|res| res.value)
}

fn macro_call_as_call_id_(
    db: &dyn ExpandDatabase,
    call: &AstIdWithPath<ast::MacroCall>,
    expand_to: ExpandTo,
    krate: CrateId,
    resolver: impl Fn(path::ModPath) -> Option<MacroDefId>,
) -> Result<ExpandResult<Option<MacroCallId>>, UnresolvedMacro> {
    let def =
        resolver(call.path.clone()).ok_or_else(|| UnresolvedMacro { path: call.path.clone() })?;

    let res = if let MacroDefKind::BuiltInEager(..) = def.kind {
        let macro_call = InFile::new(call.ast_id.file_id, call.ast_id.to_node(db));
        expand_eager_macro(db, krate, macro_call, def, &resolver)?
    } else {
        ExpandResult {
            value: Some(def.as_lazy_macro(
                db,
                krate,
                MacroCallKind::FnLike { ast_id: call.ast_id, expand_to },
            )),
            err: None,
        }
    };
    Ok(res)
}

pub fn macro_id_to_def_id(db: &dyn db::DefDatabase, id: MacroId) -> MacroDefId {
    match id {
        MacroId::Macro2Id(it) => {
            let loc = it.lookup(db);

            let item_tree = loc.id.item_tree(db);
            let makro = &item_tree[loc.id.value];
            let in_file = |m: FileAstId<ast::MacroDef>| InFile::new(loc.id.file_id(), m.upcast());
            MacroDefId {
                krate: loc.container.krate,
                kind: match loc.expander {
                    MacroExpander::Declarative => MacroDefKind::Declarative(in_file(makro.ast_id)),
                    MacroExpander::BuiltIn(it) => MacroDefKind::BuiltIn(it, in_file(makro.ast_id)),
                    MacroExpander::BuiltInAttr(it) => {
                        MacroDefKind::BuiltInAttr(it, in_file(makro.ast_id))
                    }
                    MacroExpander::BuiltInDerive(it) => {
                        MacroDefKind::BuiltInDerive(it, in_file(makro.ast_id))
                    }
                    MacroExpander::BuiltInEager(it) => {
                        MacroDefKind::BuiltInEager(it, in_file(makro.ast_id))
                    }
                },
                local_inner: false,
                allow_internal_unsafe: loc.allow_internal_unsafe,
            }
        }
        MacroId::MacroRulesId(it) => {
            let loc = it.lookup(db);

            let item_tree = loc.id.item_tree(db);
            let makro = &item_tree[loc.id.value];
            let in_file = |m: FileAstId<ast::MacroRules>| InFile::new(loc.id.file_id(), m.upcast());
            MacroDefId {
                krate: loc.container.krate,
                kind: match loc.expander {
                    MacroExpander::Declarative => MacroDefKind::Declarative(in_file(makro.ast_id)),
                    MacroExpander::BuiltIn(it) => MacroDefKind::BuiltIn(it, in_file(makro.ast_id)),
                    MacroExpander::BuiltInAttr(it) => {
                        MacroDefKind::BuiltInAttr(it, in_file(makro.ast_id))
                    }
                    MacroExpander::BuiltInDerive(it) => {
                        MacroDefKind::BuiltInDerive(it, in_file(makro.ast_id))
                    }
                    MacroExpander::BuiltInEager(it) => {
                        MacroDefKind::BuiltInEager(it, in_file(makro.ast_id))
                    }
                },
                local_inner: loc.local_inner,
                allow_internal_unsafe: loc.allow_internal_unsafe,
            }
        }
        MacroId::ProcMacroId(it) => {
            let loc = it.lookup(db);

            let item_tree = loc.id.item_tree(db);
            let makro = &item_tree[loc.id.value];
            MacroDefId {
                krate: loc.container.krate,
                kind: MacroDefKind::ProcMacro(
                    loc.expander,
                    loc.kind,
                    InFile::new(loc.id.file_id(), makro.ast_id),
                ),
                local_inner: false,
                allow_internal_unsafe: false,
            }
        }
    }
}

fn derive_macro_as_call_id(
    db: &dyn db::DefDatabase,
    item_attr: &AstIdWithPath<ast::Adt>,
    derive_attr_index: AttrId,
    derive_pos: u32,
    krate: CrateId,
    resolver: impl Fn(path::ModPath) -> Option<(MacroId, MacroDefId)>,
) -> Result<(MacroId, MacroDefId, MacroCallId), UnresolvedMacro> {
    let (macro_id, def_id) = resolver(item_attr.path.clone())
        .ok_or_else(|| UnresolvedMacro { path: item_attr.path.clone() })?;
    let call_id = def_id.as_lazy_macro(
        db.upcast(),
        krate,
        MacroCallKind::Derive {
            ast_id: item_attr.ast_id,
            derive_index: derive_pos,
            derive_attr_index,
        },
    );
    Ok((macro_id, def_id, call_id))
}

fn attr_macro_as_call_id(
    db: &dyn db::DefDatabase,
    item_attr: &AstIdWithPath<ast::Item>,
    macro_attr: &Attr,
    krate: CrateId,
    def: MacroDefId,
) -> MacroCallId {
    let arg = match macro_attr.input.as_deref() {
        Some(AttrInput::TokenTree(tt, map)) => (
            {
                let mut tt = tt.clone();
                tt.delimiter = tt::Delimiter::UNSPECIFIED;
                tt
            },
            map.clone(),
        ),
        _ => (tt::Subtree::empty(), Default::default()),
    };

    def.as_lazy_macro(
        db.upcast(),
        krate,
        MacroCallKind::Attr {
            ast_id: item_attr.ast_id,
            attr_args: Arc::new(arg),
            invoc_attr_index: macro_attr.id,
        },
    )
}
intern::impl_internable!(
    crate::type_ref::TypeRef,
    crate::type_ref::TraitRef,
    crate::type_ref::TypeBound,
    crate::path::GenericArgs,
    generics::GenericParams,
);
