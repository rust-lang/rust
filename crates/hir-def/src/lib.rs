//! `hir_def` crate contains everything between macro expansion and type
//! inference.
//!
//! It defines various items (structs, enums, traits) which comprises Rust code,
//! as well as an algorithm for resolving paths to such entities.
//!
//! Note that `hir_def` is a work in progress, so not all of the above is
//! actually true.

#![warn(rust_2018_idioms, unused_lifetimes)]
#![cfg_attr(feature = "in-rust-tree", feature(rustc_private))]

#[cfg(feature = "in-rust-tree")]
extern crate rustc_parse_format;

#[cfg(not(feature = "in-rust-tree"))]
extern crate ra_ap_rustc_parse_format as rustc_parse_format;

#[cfg(feature = "in-rust-tree")]
extern crate rustc_abi;

#[cfg(not(feature = "in-rust-tree"))]
extern crate ra_ap_rustc_abi as rustc_abi;

pub mod db;

pub mod attr;
pub mod builtin_type;
pub mod item_scope;
pub mod path;
pub mod per_ns;

pub mod expander;
pub mod lower;

pub mod dyn_map;

pub mod item_tree;

pub mod data;
pub mod generics;
pub mod lang_item;

pub mod hir;
pub use self::hir::type_ref;
pub mod body;
pub mod resolver;

pub mod nameres;
mod trace;

pub mod child_by_source;
pub mod src;

pub mod find_path;
pub mod import_map;
pub mod visibility;

pub use rustc_abi as layout;
use triomphe::Arc;

#[cfg(test)]
mod macro_expansion_tests;
mod pretty;
#[cfg(test)]
mod test_db;

use std::{
    hash::{Hash, Hasher},
    panic::{RefUnwindSafe, UnwindSafe},
};

use base_db::{
    impl_intern_key,
    salsa::{self, impl_intern_value_trivial},
    CrateId, Edition,
};
use hir_expand::{
    builtin_attr_macro::BuiltinAttrExpander,
    builtin_derive_macro::BuiltinDeriveExpander,
    builtin_fn_macro::{BuiltinFnLikeExpander, EagerExpander},
    db::ExpandDatabase,
    eager::expand_eager_macro_input,
    impl_intern_lookup,
    name::Name,
    proc_macro::{CustomProcMacroExpander, ProcMacroKind},
    AstId, ExpandError, ExpandResult, ExpandTo, HirFileId, InFile, MacroCallId, MacroCallKind,
    MacroDefId, MacroDefKind,
};
use item_tree::ExternBlock;
use la_arena::Idx;
use nameres::DefMap;
use span::{AstIdNode, FileAstId, FileId, Span};
use stdx::impl_from;
use syntax::{ast, AstNode};

pub use hir_expand::{tt, Intern, Lookup};

use crate::{
    builtin_type::BuiltinType,
    data::adt::VariantData,
    db::DefDatabase,
    item_tree::{
        Const, Enum, ExternCrate, Function, Impl, ItemTreeId, ItemTreeNode, Macro2, MacroRules,
        Static, Struct, Trait, TraitAlias, TypeAlias, Union, Use, Variant,
    },
};

#[derive(Debug)]
pub struct ItemLoc<N: ItemTreeNode> {
    pub container: ModuleId,
    pub id: ItemTreeId<N>,
}

impl<N: ItemTreeNode> Clone for ItemLoc<N> {
    fn clone(&self) -> Self {
        *self
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
        *self
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

pub trait ItemTreeLoc {
    type Container;
    type Id;
    fn item_tree_id(&self) -> ItemTreeId<Self::Id>;
    fn container(&self) -> Self::Container;
}

macro_rules! impl_intern {
    ($id:ident, $loc:ident, $intern:ident, $lookup:ident) => {
        impl_intern_key!($id);
        impl_intern_value_trivial!($loc);
        impl_intern_lookup!(DefDatabase, $id, $loc, $intern, $lookup);
    };
}

macro_rules! impl_loc {
    ($loc:ident, $id:ident: $id_ty:ident, $container:ident: $container_type:ident) => {
        impl ItemTreeLoc for $loc {
            type Container = $container_type;
            type Id = $id_ty;
            fn item_tree_id(&self) -> ItemTreeId<Self::Id> {
                self.$id
            }
            fn container(&self) -> Self::Container {
                self.$container
            }
        }
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FunctionId(salsa::InternId);
type FunctionLoc = AssocItemLoc<Function>;
impl_intern!(FunctionId, FunctionLoc, intern_function, lookup_intern_function);
impl_loc!(FunctionLoc, id: Function, container: ItemContainerId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StructId(salsa::InternId);
type StructLoc = ItemLoc<Struct>;
impl_intern!(StructId, StructLoc, intern_struct, lookup_intern_struct);
impl_loc!(StructLoc, id: Struct, container: ModuleId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct UnionId(salsa::InternId);
pub type UnionLoc = ItemLoc<Union>;
impl_intern!(UnionId, UnionLoc, intern_union, lookup_intern_union);
impl_loc!(UnionLoc, id: Union, container: ModuleId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EnumId(salsa::InternId);
pub type EnumLoc = ItemLoc<Enum>;
impl_intern!(EnumId, EnumLoc, intern_enum, lookup_intern_enum);
impl_loc!(EnumLoc, id: Enum, container: ModuleId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConstId(salsa::InternId);
type ConstLoc = AssocItemLoc<Const>;
impl_intern!(ConstId, ConstLoc, intern_const, lookup_intern_const);
impl_loc!(ConstLoc, id: Const, container: ItemContainerId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StaticId(salsa::InternId);
pub type StaticLoc = AssocItemLoc<Static>;
impl_intern!(StaticId, StaticLoc, intern_static, lookup_intern_static);
impl_loc!(StaticLoc, id: Static, container: ItemContainerId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TraitId(salsa::InternId);
pub type TraitLoc = ItemLoc<Trait>;
impl_intern!(TraitId, TraitLoc, intern_trait, lookup_intern_trait);
impl_loc!(TraitLoc, id: Trait, container: ModuleId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TraitAliasId(salsa::InternId);
pub type TraitAliasLoc = ItemLoc<TraitAlias>;
impl_intern!(TraitAliasId, TraitAliasLoc, intern_trait_alias, lookup_intern_trait_alias);
impl_loc!(TraitAliasLoc, id: TraitAlias, container: ModuleId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeAliasId(salsa::InternId);
type TypeAliasLoc = AssocItemLoc<TypeAlias>;
impl_intern!(TypeAliasId, TypeAliasLoc, intern_type_alias, lookup_intern_type_alias);
impl_loc!(TypeAliasLoc, id: TypeAlias, container: ItemContainerId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct ImplId(salsa::InternId);
type ImplLoc = ItemLoc<Impl>;
impl_intern!(ImplId, ImplLoc, intern_impl, lookup_intern_impl);
impl_loc!(ImplLoc, id: Impl, container: ModuleId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct UseId(salsa::InternId);
type UseLoc = ItemLoc<Use>;
impl_intern!(UseId, UseLoc, intern_use, lookup_intern_use);
impl_loc!(UseLoc, id: Use, container: ModuleId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct ExternCrateId(salsa::InternId);
type ExternCrateLoc = ItemLoc<ExternCrate>;
impl_intern!(ExternCrateId, ExternCrateLoc, intern_extern_crate, lookup_intern_extern_crate);
impl_loc!(ExternCrateLoc, id: ExternCrate, container: ModuleId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct ExternBlockId(salsa::InternId);
type ExternBlockLoc = ItemLoc<ExternBlock>;
impl_intern!(ExternBlockId, ExternBlockLoc, intern_extern_block, lookup_intern_extern_block);
impl_loc!(ExternBlockLoc, id: ExternBlock, container: ModuleId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EnumVariantId(salsa::InternId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EnumVariantLoc {
    pub id: ItemTreeId<Variant>,
    pub parent: EnumId,
    pub index: u32,
}
impl_intern!(EnumVariantId, EnumVariantLoc, intern_enum_variant, lookup_intern_enum_variant);
impl_loc!(EnumVariantLoc, id: Variant, parent: EnumId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct Macro2Id(salsa::InternId);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Macro2Loc {
    pub container: ModuleId,
    pub id: ItemTreeId<Macro2>,
    pub expander: MacroExpander,
    pub allow_internal_unsafe: bool,
    pub edition: Edition,
}
impl_intern!(Macro2Id, Macro2Loc, intern_macro2, lookup_intern_macro2);
impl_loc!(Macro2Loc, id: Macro2, container: ModuleId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct MacroRulesId(salsa::InternId);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MacroRulesLoc {
    pub container: ModuleId,
    pub id: ItemTreeId<MacroRules>,
    pub expander: MacroExpander,
    pub flags: MacroRulesLocFlags,
    pub edition: Edition,
}
impl_intern!(MacroRulesId, MacroRulesLoc, intern_macro_rules, lookup_intern_macro_rules);
impl_loc!(MacroRulesLoc, id: MacroRules, container: ModuleId);

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct MacroRulesLocFlags: u8 {
        const ALLOW_INTERNAL_UNSAFE = 1 << 0;
        const LOCAL_INNER = 1 << 1;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MacroExpander {
    Declarative,
    BuiltIn(BuiltinFnLikeExpander),
    BuiltInAttr(BuiltinAttrExpander),
    BuiltInDerive(BuiltinDeriveExpander),
    BuiltInEager(EagerExpander),
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct ProcMacroId(salsa::InternId);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProcMacroLoc {
    pub container: CrateRootModuleId,
    pub id: ItemTreeId<Function>,
    pub expander: CustomProcMacroExpander,
    pub kind: ProcMacroKind,
    pub edition: Edition,
}
impl_intern!(ProcMacroId, ProcMacroLoc, intern_proc_macro, lookup_intern_proc_macro);
impl_loc!(ProcMacroLoc, id: Function, container: CrateRootModuleId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct BlockId(salsa::InternId);
#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub struct BlockLoc {
    ast_id: AstId<ast::BlockExpr>,
    /// The containing module.
    module: ModuleId,
}
impl_intern!(BlockId, BlockLoc, intern_block, lookup_intern_block);

/// Id of the anonymous const block expression and patterns. This is very similar to `ClosureId` and
/// shouldn't be a `DefWithBodyId` since its type inference is dependent on its parent.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct ConstBlockId(salsa::InternId);
impl_intern!(ConstBlockId, ConstBlockLoc, intern_anonymous_const, lookup_intern_anonymous_const);

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub struct ConstBlockLoc {
    /// The parent of the anonymous const block.
    pub parent: DefWithBodyId,
    /// The root expression of this const block in the parent body.
    pub root: hir::ExprId,
}

/// A `ModuleId` that is always a crate's root module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CrateRootModuleId {
    krate: CrateId,
}

impl CrateRootModuleId {
    pub fn def_map(&self, db: &dyn DefDatabase) -> Arc<DefMap> {
        db.crate_def_map(self.krate)
    }

    pub fn krate(self) -> CrateId {
        self.krate
    }
}

impl PartialEq<ModuleId> for CrateRootModuleId {
    fn eq(&self, other: &ModuleId) -> bool {
        other.block.is_none() && other.local_id == DefMap::ROOT && self.krate == other.krate
    }
}

impl From<CrateId> for CrateRootModuleId {
    fn from(krate: CrateId) -> Self {
        CrateRootModuleId { krate }
    }
}

impl TryFrom<ModuleId> for CrateRootModuleId {
    type Error = ();

    fn try_from(ModuleId { krate, block, local_id }: ModuleId) -> Result<Self, Self::Error> {
        if block.is_none() && local_id == DefMap::ROOT {
            Ok(CrateRootModuleId { krate })
        } else {
            Err(())
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
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
    pub fn def_map(self, db: &dyn DefDatabase) -> Arc<DefMap> {
        match self.block {
            Some(block) => db.block_def_map(block),
            None => db.crate_def_map(self.krate),
        }
    }

    pub fn krate(self) -> CrateId {
        self.krate
    }

    pub fn name(self, db: &dyn DefDatabase) -> Option<Name> {
        let def_map = self.def_map(db);
        let parent = def_map[self.local_id].parent?;
        def_map[parent].children.iter().find_map(|(name, module_id)| {
            if *module_id == self.local_id {
                Some(name.clone())
            } else {
                None
            }
        })
    }

    pub fn containing_module(self, db: &dyn DefDatabase) -> Option<ModuleId> {
        self.def_map(db).containing_module(self.local_id)
    }

    pub fn containing_block(self) -> Option<BlockId> {
        self.block
    }

    pub fn is_block_module(self) -> bool {
        self.block.is_some() && self.local_id == DefMap::ROOT
    }
}

impl PartialEq<CrateRootModuleId> for ModuleId {
    fn eq(&self, other: &CrateRootModuleId) -> bool {
        other == self
    }
}

impl From<CrateRootModuleId> for ModuleId {
    fn from(CrateRootModuleId { krate }: CrateRootModuleId) -> Self {
        ModuleId { krate, block: None, local_id: DefMap::ROOT }
    }
}

impl From<CrateRootModuleId> for ModuleDefId {
    fn from(value: CrateRootModuleId) -> Self {
        ModuleDefId::ModuleId(value.into())
    }
}

/// An ID of a module, **local** to a `DefMap`.
pub type LocalModuleId = Idx<nameres::ModuleData>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FieldId {
    pub parent: VariantId,
    pub local_id: LocalFieldId,
}

pub type LocalFieldId = Idx<data::adt::FieldData>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TupleId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TupleFieldId {
    pub tuple: TupleId,
    pub index: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeOrConstParamId {
    pub parent: GenericDefId,
    pub local_id: LocalTypeOrConstParamId,
}
impl_intern_value_trivial!(TypeOrConstParamId);

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
    pub fn from_unchecked(it: TypeOrConstParamId) -> Self {
        Self(it)
    }
}

impl From<TypeParamId> for TypeOrConstParamId {
    fn from(it: TypeParamId) -> Self {
        it.0
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
    pub fn from_unchecked(it: TypeOrConstParamId) -> Self {
        Self(it)
    }
}

impl From<ConstParamId> for TypeOrConstParamId {
    fn from(it: ConstParamId) -> Self {
        it.0
    }
}

pub type LocalTypeOrConstParamId = Idx<generics::TypeOrConstParamData>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LifetimeParamId {
    pub parent: GenericDefId,
    pub local_id: LocalLifetimeParamId,
}
pub type LocalLifetimeParamId = Idx<generics::LifetimeParamData>;
impl_intern_value_trivial!(LifetimeParamId);

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
    pub fn is_attribute(self, db: &dyn DefDatabase) -> bool {
        matches!(self, MacroId::ProcMacroId(it) if it.lookup(db).kind == ProcMacroKind::Attr)
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

/// Something that holds types, required for the current const arg lowering implementation as they
/// need to be able to query where they are defined.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum TypeOwnerId {
    FunctionId(FunctionId),
    StaticId(StaticId),
    ConstId(ConstId),
    InTypeConstId(InTypeConstId),
    AdtId(AdtId),
    TraitId(TraitId),
    TraitAliasId(TraitAliasId),
    TypeAliasId(TypeAliasId),
    ImplId(ImplId),
    EnumVariantId(EnumVariantId),
}

impl TypeOwnerId {
    fn as_generic_def_id(self) -> Option<GenericDefId> {
        Some(match self {
            TypeOwnerId::FunctionId(it) => GenericDefId::FunctionId(it),
            TypeOwnerId::ConstId(it) => GenericDefId::ConstId(it),
            TypeOwnerId::AdtId(it) => GenericDefId::AdtId(it),
            TypeOwnerId::TraitId(it) => GenericDefId::TraitId(it),
            TypeOwnerId::TraitAliasId(it) => GenericDefId::TraitAliasId(it),
            TypeOwnerId::TypeAliasId(it) => GenericDefId::TypeAliasId(it),
            TypeOwnerId::ImplId(it) => GenericDefId::ImplId(it),
            TypeOwnerId::EnumVariantId(it) => GenericDefId::EnumVariantId(it),
            TypeOwnerId::InTypeConstId(_) | TypeOwnerId::StaticId(_) => return None,
        })
    }
}

impl_from!(
    FunctionId,
    StaticId,
    ConstId,
    InTypeConstId,
    AdtId,
    TraitId,
    TraitAliasId,
    TypeAliasId,
    ImplId,
    EnumVariantId
    for TypeOwnerId
);

// Every `DefWithBodyId` is a type owner, since bodies can contain type (e.g. `{ let it: Type = _; }`)
impl From<DefWithBodyId> for TypeOwnerId {
    fn from(value: DefWithBodyId) -> Self {
        match value {
            DefWithBodyId::FunctionId(it) => it.into(),
            DefWithBodyId::StaticId(it) => it.into(),
            DefWithBodyId::ConstId(it) => it.into(),
            DefWithBodyId::InTypeConstId(it) => it.into(),
            DefWithBodyId::VariantId(it) => it.into(),
        }
    }
}

impl From<GenericDefId> for TypeOwnerId {
    fn from(value: GenericDefId) -> Self {
        match value {
            GenericDefId::FunctionId(it) => it.into(),
            GenericDefId::AdtId(it) => it.into(),
            GenericDefId::TraitId(it) => it.into(),
            GenericDefId::TraitAliasId(it) => it.into(),
            GenericDefId::TypeAliasId(it) => it.into(),
            GenericDefId::ImplId(it) => it.into(),
            GenericDefId::EnumVariantId(it) => it.into(),
            GenericDefId::ConstId(it) => it.into(),
        }
    }
}

// FIXME: This should not be a thing
/// A thing that we want to store in interned ids, but we don't know its type in `hir-def`. This is
/// currently only used in `InTypeConstId` for storing the type (which has type `Ty` defined in
/// the `hir-ty` crate) of the constant in its id, which is a temporary hack so we may want
/// to remove this after removing that.
pub trait OpaqueInternableThing:
    std::any::Any + std::fmt::Debug + Sync + Send + UnwindSafe + RefUnwindSafe
{
    fn as_any(&self) -> &dyn std::any::Any;
    fn box_any(&self) -> Box<dyn std::any::Any>;
    fn dyn_hash(&self, state: &mut dyn Hasher);
    fn dyn_eq(&self, other: &dyn OpaqueInternableThing) -> bool;
    fn dyn_clone(&self) -> Box<dyn OpaqueInternableThing>;
}

impl Hash for dyn OpaqueInternableThing {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.dyn_hash(state);
    }
}

impl PartialEq for dyn OpaqueInternableThing {
    fn eq(&self, other: &Self) -> bool {
        self.dyn_eq(other)
    }
}

impl Eq for dyn OpaqueInternableThing {}

impl Clone for Box<dyn OpaqueInternableThing> {
    fn clone(&self) -> Self {
        self.dyn_clone()
    }
}

// FIXME(const-generic-body): Use an stable id for in type consts.
//
// The current id uses `AstId<ast::ConstArg>` which will be changed by every change in the code. Ideally
// we should use an id which is relative to the type owner, so that every change will only invalidate the
// id if it happens inside of the type owner.
//
// The solution probably is to have some query on `TypeOwnerId` to traverse its constant children and store
// their `AstId` in a list (vector or arena), and use the index of that list in the id here. That query probably
// needs name resolution, and might go far and handles the whole path lowering or type lowering for a `TypeOwnerId`.
//
// Whatever path the solution takes, it should answer 3 questions at the same time:
// * Is the id stable enough?
// * How to find a constant id using an ast node / position in the source code? This is needed when we want to
//   provide ide functionalities inside an in type const (which we currently don't support) e.g. go to definition
//   for a local defined there. A complex id might have some trouble in this reverse mapping.
// * How to find the return type of a constant using its id? We have this data when we are doing type lowering
//   and the name of the struct that contains this constant is resolved, so a query that only traverses the
//   type owner by its syntax tree might have a hard time here.

/// A constant in a type as a substitution for const generics (like `Foo<{ 2 + 2 }>`) or as an array
/// length (like `[u8; 2 + 2]`). These constants are body owner and are a variant of `DefWithBodyId`. These
/// are not called `AnonymousConstId` to prevent confusion with [`ConstBlockId`].
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct InTypeConstId(salsa::InternId);
impl_intern!(InTypeConstId, InTypeConstLoc, intern_in_type_const, lookup_intern_in_type_const);

// We would like to set `derive(PartialEq)`
// but the compiler complains about that `.expected_ty` does not implement the `Copy` trait.
#[allow(clippy::derived_hash_with_manual_eq)]
#[derive(Debug, Hash, Eq, Clone)]
pub struct InTypeConstLoc {
    pub id: AstId<ast::ConstArg>,
    /// The thing this const arg appears in
    pub owner: TypeOwnerId,
    // FIXME(const-generic-body): The expected type should not be
    pub expected_ty: Box<dyn OpaqueInternableThing>,
}

impl PartialEq for InTypeConstLoc {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.owner == other.owner && *self.expected_ty == *other.expected_ty
    }
}

impl InTypeConstId {
    pub fn source(&self, db: &dyn DefDatabase) -> ast::ConstArg {
        let src = self.lookup(db).id;
        let file_id = src.file_id;
        let root = &db.parse_or_expand(file_id);
        db.ast_id_map(file_id).get(src.value).to_node(root)
    }
}

/// A constant, which might appears as a const item, an anonymous const block in expressions
/// or patterns, or as a constant in types with const generics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GeneralConstId {
    ConstId(ConstId),
    ConstBlockId(ConstBlockId),
    InTypeConstId(InTypeConstId),
}

impl_from!(ConstId, ConstBlockId, InTypeConstId for GeneralConstId);

impl GeneralConstId {
    pub fn generic_def(self, db: &dyn DefDatabase) -> Option<GenericDefId> {
        match self {
            GeneralConstId::ConstId(it) => Some(it.into()),
            GeneralConstId::ConstBlockId(it) => it.lookup(db).parent.as_generic_def_id(),
            GeneralConstId::InTypeConstId(it) => it.lookup(db).owner.as_generic_def_id(),
        }
    }

    pub fn name(self, db: &dyn DefDatabase) -> String {
        match self {
            GeneralConstId::ConstId(const_id) => db
                .const_data(const_id)
                .name
                .as_ref()
                .and_then(|it| it.as_str())
                .unwrap_or("_")
                .to_owned(),
            GeneralConstId::ConstBlockId(id) => format!("{{anonymous const {id:?}}}"),
            GeneralConstId::InTypeConstId(id) => format!("{{in type const {id:?}}}"),
        }
    }
}

/// The defs which have a body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DefWithBodyId {
    FunctionId(FunctionId),
    StaticId(StaticId),
    ConstId(ConstId),
    InTypeConstId(InTypeConstId),
    VariantId(EnumVariantId),
}

impl_from!(FunctionId, ConstId, StaticId, InTypeConstId for DefWithBodyId);

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
            // FIXME: stable rust doesn't allow generics in constants, but we should
            // use `TypeOwnerId::as_generic_def_id` when it does.
            DefWithBodyId::InTypeConstId(_) => None,
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

impl GenericDefId {
    fn file_id_and_params_of(
        self,
        db: &dyn DefDatabase,
    ) -> (HirFileId, Option<ast::GenericParamList>) {
        fn file_id_and_params_of_item_loc<Loc>(
            db: &dyn DefDatabase,
            def: impl for<'db> Lookup<Database<'db> = dyn DefDatabase + 'db, Data = Loc>,
        ) -> (HirFileId, Option<ast::GenericParamList>)
        where
            Loc: src::HasSource,
            Loc::Value: ast::HasGenericParams,
        {
            let src = def.lookup(db).source(db);
            (src.file_id, ast::HasGenericParams::generic_param_list(&src.value))
        }

        match self {
            GenericDefId::FunctionId(it) => file_id_and_params_of_item_loc(db, it),
            GenericDefId::TypeAliasId(it) => file_id_and_params_of_item_loc(db, it),
            GenericDefId::ConstId(_) => (FileId::BOGUS.into(), None),
            GenericDefId::AdtId(AdtId::StructId(it)) => file_id_and_params_of_item_loc(db, it),
            GenericDefId::AdtId(AdtId::UnionId(it)) => file_id_and_params_of_item_loc(db, it),
            GenericDefId::AdtId(AdtId::EnumId(it)) => file_id_and_params_of_item_loc(db, it),
            GenericDefId::TraitId(it) => file_id_and_params_of_item_loc(db, it),
            GenericDefId::TraitAliasId(it) => file_id_and_params_of_item_loc(db, it),
            GenericDefId::ImplId(it) => file_id_and_params_of_item_loc(db, it),
            // We won't be using this ID anyway
            GenericDefId::EnumVariantId(_) => (FileId::BOGUS.into(), None),
        }
    }
}

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
    ExternCrateId(ExternCrateId),
    UseId(UseId),
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
    TraitAliasId,
    TypeAliasId,
    MacroId(Macro2Id, MacroRulesId, ProcMacroId),
    ImplId,
    GenericParamId,
    ExternCrateId,
    UseId
    for AttrDefId
);

impl TryFrom<ModuleDefId> for AttrDefId {
    type Error = ();

    fn try_from(value: ModuleDefId) -> Result<Self, Self::Error> {
        match value {
            ModuleDefId::ModuleId(it) => Ok(it.into()),
            ModuleDefId::FunctionId(it) => Ok(it.into()),
            ModuleDefId::AdtId(it) => Ok(it.into()),
            ModuleDefId::EnumVariantId(it) => Ok(it.into()),
            ModuleDefId::ConstId(it) => Ok(it.into()),
            ModuleDefId::StaticId(it) => Ok(it.into()),
            ModuleDefId::TraitId(it) => Ok(it.into()),
            ModuleDefId::TypeAliasId(it) => Ok(it.into()),
            ModuleDefId::TraitAliasId(id) => Ok(id.into()),
            ModuleDefId::MacroId(id) => Ok(id.into()),
            ModuleDefId::BuiltinType(_) => Err(()),
        }
    }
}

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
impl From<AssocItemId> for AttrDefId {
    fn from(assoc: AssocItemId) -> Self {
        match assoc {
            AssocItemId::FunctionId(it) => AttrDefId::FunctionId(it),
            AssocItemId::ConstId(it) => AttrDefId::ConstId(it),
            AssocItemId::TypeAliasId(it) => AttrDefId::TypeAliasId(it),
        }
    }
}
impl From<VariantId> for AttrDefId {
    fn from(vid: VariantId) -> Self {
        match vid {
            VariantId::EnumVariantId(id) => id.into(),
            VariantId::StructId(id) => id.into(),
            VariantId::UnionId(id) => id.into(),
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
    pub fn variant_data(self, db: &dyn DefDatabase) -> Arc<VariantData> {
        match self {
            VariantId::StructId(it) => db.struct_data(it).variant_data.clone(),
            VariantId::UnionId(it) => db.union_data(it).variant_data.clone(),
            VariantId::EnumVariantId(it) => db.enum_variant_data(it).variant_data.clone(),
        }
    }

    pub fn file_id(self, db: &dyn DefDatabase) -> HirFileId {
        match self {
            VariantId::EnumVariantId(it) => it.lookup(db).id.file_id(),
            VariantId::StructId(it) => it.lookup(db).id.file_id(),
            VariantId::UnionId(it) => it.lookup(db).id.file_id(),
        }
    }

    pub fn adt_id(self, db: &dyn DefDatabase) -> AdtId {
        match self {
            VariantId::EnumVariantId(it) => it.lookup(db).parent.into(),
            VariantId::StructId(it) => it.into(),
            VariantId::UnionId(it) => it.into(),
        }
    }
}

pub trait HasModule {
    /// Returns the enclosing module this thing is defined within.
    fn module(&self, db: &dyn DefDatabase) -> ModuleId;
    /// Returns the crate this thing is defined within.
    #[inline]
    #[doc(alias = "crate")]
    fn krate(&self, db: &dyn DefDatabase) -> CrateId {
        self.module(db).krate
    }
}

// In theory this impl should work out for us, but rustc thinks it collides with all the other
// manual impls that do not have a ModuleId container...
// impl<N, ItemId, Data> HasModule for ItemId
// where
//     N: ItemTreeNode,
//     ItemId: for<'db> Lookup<Database<'db> = dyn DefDatabase + 'db, Data = Data> + Copy,
//     Data: ItemTreeLoc<Id = N, Container = ModuleId>,
// {
//     #[inline]
//     fn module(&self, db: &dyn DefDatabase) -> ModuleId {
//         self.lookup(db).container()
//     }
// }

impl<N, ItemId> HasModule for ItemId
where
    N: ItemTreeNode,
    ItemId: for<'db> Lookup<Database<'db> = dyn DefDatabase + 'db, Data = ItemLoc<N>> + Copy,
{
    #[inline]
    fn module(&self, db: &dyn DefDatabase) -> ModuleId {
        self.lookup(db).container
    }
}

// Technically this does not overlap with the above, but rustc currently forbids this, hence why we
// need to write the 3 impls manually instead
// impl<N, ItemId> HasModule for ItemId
// where
//     N: ItemTreeModItemNode,
//     ItemId: for<'db> Lookup<Database<'db> = dyn DefDatabase + 'db, Data = AssocItemLoc<N>> + Copy,
// {
//     #[inline]
//     fn module(&self, db: &dyn DefDatabase) -> ModuleId {
//         self.lookup(db).container.module(db)
//     }
// }

// region: manual-assoc-has-module-impls
#[inline]
fn module_for_assoc_item_loc<'db>(
    db: &(dyn 'db + DefDatabase),
    id: impl Lookup<Database<'db> = dyn DefDatabase + 'db, Data = AssocItemLoc<impl ItemTreeNode>>,
) -> ModuleId {
    id.lookup(db).container.module(db)
}

impl HasModule for FunctionId {
    #[inline]
    fn module(&self, db: &dyn DefDatabase) -> ModuleId {
        module_for_assoc_item_loc(db, *self)
    }
}

impl HasModule for ConstId {
    #[inline]
    fn module(&self, db: &dyn DefDatabase) -> ModuleId {
        module_for_assoc_item_loc(db, *self)
    }
}

impl HasModule for StaticId {
    #[inline]
    fn module(&self, db: &dyn DefDatabase) -> ModuleId {
        module_for_assoc_item_loc(db, *self)
    }
}

impl HasModule for TypeAliasId {
    #[inline]
    fn module(&self, db: &dyn DefDatabase) -> ModuleId {
        module_for_assoc_item_loc(db, *self)
    }
}
// endregion: manual-assoc-has-module-impls

impl HasModule for EnumVariantId {
    #[inline]
    fn module(&self, db: &dyn DefDatabase) -> ModuleId {
        self.lookup(db).parent.module(db)
    }
}

impl HasModule for MacroRulesId {
    #[inline]
    fn module(&self, db: &dyn DefDatabase) -> ModuleId {
        self.lookup(db).container
    }
}

impl HasModule for Macro2Id {
    #[inline]
    fn module(&self, db: &dyn DefDatabase) -> ModuleId {
        self.lookup(db).container
    }
}

impl HasModule for ProcMacroId {
    #[inline]
    fn module(&self, db: &dyn DefDatabase) -> ModuleId {
        self.lookup(db).container.into()
    }
}

impl HasModule for ItemContainerId {
    fn module(&self, db: &dyn DefDatabase) -> ModuleId {
        match *self {
            ItemContainerId::ModuleId(it) => it,
            ItemContainerId::ImplId(it) => it.module(db),
            ItemContainerId::TraitId(it) => it.module(db),
            ItemContainerId::ExternBlockId(it) => it.module(db),
        }
    }
}

impl HasModule for AdtId {
    fn module(&self, db: &dyn DefDatabase) -> ModuleId {
        match *self {
            AdtId::StructId(it) => it.module(db),
            AdtId::UnionId(it) => it.module(db),
            AdtId::EnumId(it) => it.module(db),
        }
    }
}

impl HasModule for VariantId {
    fn module(&self, db: &dyn DefDatabase) -> ModuleId {
        match *self {
            VariantId::EnumVariantId(it) => it.module(db),
            VariantId::StructId(it) => it.module(db),
            VariantId::UnionId(it) => it.module(db),
        }
    }
}

impl HasModule for MacroId {
    fn module(&self, db: &dyn DefDatabase) -> ModuleId {
        match *self {
            MacroId::MacroRulesId(it) => it.module(db),
            MacroId::Macro2Id(it) => it.module(db),
            MacroId::ProcMacroId(it) => it.module(db),
        }
    }
}

impl HasModule for TypeOwnerId {
    fn module(&self, db: &dyn DefDatabase) -> ModuleId {
        match *self {
            TypeOwnerId::FunctionId(it) => it.module(db),
            TypeOwnerId::StaticId(it) => it.module(db),
            TypeOwnerId::ConstId(it) => it.module(db),
            TypeOwnerId::AdtId(it) => it.module(db),
            TypeOwnerId::TraitId(it) => it.module(db),
            TypeOwnerId::TraitAliasId(it) => it.module(db),
            TypeOwnerId::TypeAliasId(it) => it.module(db),
            TypeOwnerId::ImplId(it) => it.module(db),
            TypeOwnerId::EnumVariantId(it) => it.module(db),
            TypeOwnerId::InTypeConstId(it) => it.lookup(db).owner.module(db),
        }
    }
}

impl HasModule for DefWithBodyId {
    fn module(&self, db: &dyn DefDatabase) -> ModuleId {
        match self {
            DefWithBodyId::FunctionId(it) => it.module(db),
            DefWithBodyId::StaticId(it) => it.module(db),
            DefWithBodyId::ConstId(it) => it.module(db),
            DefWithBodyId::VariantId(it) => it.module(db),
            DefWithBodyId::InTypeConstId(it) => it.lookup(db).owner.module(db),
        }
    }
}

impl HasModule for GenericDefId {
    fn module(&self, db: &dyn DefDatabase) -> ModuleId {
        match self {
            GenericDefId::FunctionId(it) => it.module(db),
            GenericDefId::AdtId(it) => it.module(db),
            GenericDefId::TraitId(it) => it.module(db),
            GenericDefId::TraitAliasId(it) => it.module(db),
            GenericDefId::TypeAliasId(it) => it.module(db),
            GenericDefId::ImplId(it) => it.module(db),
            GenericDefId::EnumVariantId(it) => it.module(db),
            GenericDefId::ConstId(it) => it.module(db),
        }
    }
}

impl HasModule for AttrDefId {
    fn module(&self, db: &dyn DefDatabase) -> ModuleId {
        match self {
            AttrDefId::ModuleId(it) => *it,
            AttrDefId::FieldId(it) => it.parent.module(db),
            AttrDefId::AdtId(it) => it.module(db),
            AttrDefId::FunctionId(it) => it.module(db),
            AttrDefId::EnumVariantId(it) => it.module(db),
            AttrDefId::StaticId(it) => it.module(db),
            AttrDefId::ConstId(it) => it.module(db),
            AttrDefId::TraitId(it) => it.module(db),
            AttrDefId::TraitAliasId(it) => it.module(db),
            AttrDefId::TypeAliasId(it) => it.module(db),
            AttrDefId::ImplId(it) => it.module(db),
            AttrDefId::ExternBlockId(it) => it.module(db),
            AttrDefId::GenericParamId(it) => match it {
                GenericParamId::TypeParamId(it) => it.parent(),
                GenericParamId::ConstParamId(it) => it.parent(),
                GenericParamId::LifetimeParamId(it) => it.parent,
            }
            .module(db),
            AttrDefId::MacroId(it) => it.module(db),
            AttrDefId::ExternCrateId(it) => it.module(db),
            AttrDefId::UseId(it) => it.module(db),
        }
    }
}

impl ModuleDefId {
    /// Returns the module containing `self` (or `self`, if `self` is itself a module).
    ///
    /// Returns `None` if `self` refers to a primitive type.
    pub fn module(&self, db: &dyn DefDatabase) -> Option<ModuleId> {
        Some(match self {
            ModuleDefId::ModuleId(id) => *id,
            ModuleDefId::FunctionId(id) => id.module(db),
            ModuleDefId::AdtId(id) => id.module(db),
            ModuleDefId::EnumVariantId(id) => id.module(db),
            ModuleDefId::ConstId(id) => id.module(db),
            ModuleDefId::StaticId(id) => id.module(db),
            ModuleDefId::TraitId(id) => id.module(db),
            ModuleDefId::TraitAliasId(id) => id.module(db),
            ModuleDefId::TypeAliasId(id) => id.module(db),
            ModuleDefId::MacroId(id) => id.module(db),
            ModuleDefId::BuiltinType(_) => return None,
        })
    }
}

/// A helper trait for converting to MacroCallId
pub trait AsMacroCall {
    fn as_call_id(
        &self,
        db: &dyn ExpandDatabase,
        krate: CrateId,
        resolver: impl Fn(path::ModPath) -> Option<MacroDefId> + Copy,
    ) -> Option<MacroCallId> {
        self.as_call_id_with_errors(db, krate, resolver).ok()?.value
    }

    fn as_call_id_with_errors(
        &self,
        db: &dyn ExpandDatabase,
        krate: CrateId,
        resolver: impl Fn(path::ModPath) -> Option<MacroDefId> + Copy,
    ) -> Result<ExpandResult<Option<MacroCallId>>, UnresolvedMacro>;
}

impl AsMacroCall for InFile<&ast::MacroCall> {
    fn as_call_id_with_errors(
        &self,
        db: &dyn ExpandDatabase,
        krate: CrateId,
        resolver: impl Fn(path::ModPath) -> Option<MacroDefId> + Copy,
    ) -> Result<ExpandResult<Option<MacroCallId>>, UnresolvedMacro> {
        let expands_to = hir_expand::ExpandTo::from_call_site(self.value);
        let ast_id = AstId::new(self.file_id, db.ast_id_map(self.file_id).ast_id(self.value));
        let span_map = db.span_map(self.file_id);
        let path = self.value.path().and_then(|path| {
            path::ModPath::from_src(db, path, &mut |range| {
                span_map.as_ref().span_for_range(range).ctx
            })
        });

        let Some(path) = path else {
            return Ok(ExpandResult::only_err(ExpandError::other("malformed macro invocation")));
        };

        let call_site = span_map.span_for_range(self.value.syntax().text_range());

        macro_call_as_call_id_with_eager(
            db,
            &AstIdWithPath::new(ast_id.file_id, ast_id.value, path),
            call_site,
            expands_to,
            krate,
            resolver,
            resolver,
        )
    }
}

/// Helper wrapper for `AstId` with `ModPath`
#[derive(Clone, Debug, Eq, PartialEq)]
struct AstIdWithPath<T: AstIdNode> {
    ast_id: AstId<T>,
    path: path::ModPath,
}

impl<T: AstIdNode> AstIdWithPath<T> {
    fn new(file_id: HirFileId, ast_id: FileAstId<T>, path: path::ModPath) -> AstIdWithPath<T> {
        AstIdWithPath { ast_id: AstId::new(file_id, ast_id), path }
    }
}

fn macro_call_as_call_id(
    db: &dyn ExpandDatabase,
    call: &AstIdWithPath<ast::MacroCall>,
    call_site: Span,
    expand_to: ExpandTo,
    krate: CrateId,
    resolver: impl Fn(path::ModPath) -> Option<MacroDefId> + Copy,
) -> Result<Option<MacroCallId>, UnresolvedMacro> {
    macro_call_as_call_id_with_eager(db, call, call_site, expand_to, krate, resolver, resolver)
        .map(|res| res.value)
}

fn macro_call_as_call_id_with_eager(
    db: &dyn ExpandDatabase,
    call: &AstIdWithPath<ast::MacroCall>,
    call_site: Span,
    expand_to: ExpandTo,
    krate: CrateId,
    resolver: impl FnOnce(path::ModPath) -> Option<MacroDefId>,
    eager_resolver: impl Fn(path::ModPath) -> Option<MacroDefId>,
) -> Result<ExpandResult<Option<MacroCallId>>, UnresolvedMacro> {
    let def =
        resolver(call.path.clone()).ok_or_else(|| UnresolvedMacro { path: call.path.clone() })?;

    let res = match def.kind {
        MacroDefKind::BuiltInEager(..) => {
            let macro_call = InFile::new(call.ast_id.file_id, call.ast_id.to_node(db));
            expand_eager_macro_input(db, krate, macro_call, def, call_site, &|path| {
                eager_resolver(path).filter(MacroDefId::is_fn_like)
            })
        }
        _ if def.is_fn_like() => ExpandResult {
            value: Some(def.as_lazy_macro(
                db,
                krate,
                MacroCallKind::FnLike { ast_id: call.ast_id, expand_to },
                call_site,
            )),
            err: None,
        },
        _ => return Err(UnresolvedMacro { path: call.path.clone() }),
    };
    Ok(res)
}

#[derive(Debug)]
pub struct UnresolvedMacro {
    pub path: hir_expand::mod_path::ModPath,
}

intern::impl_internable!(
    crate::type_ref::TypeRef,
    crate::type_ref::TraitRef,
    crate::type_ref::TypeBound,
    crate::path::GenericArgs,
    generics::GenericParams,
);
