//! `hir_def` crate contains everything between macro expansion and type
//! inference.
//!
//! It defines various items (structs, enums, traits) which comprises Rust code,
//! as well as an algorithm for resolving paths to such entities.
//!
//! Note that `hir_def` is a work in progress, so not all of the above is
//! actually true.

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
pub mod per_ns;

pub mod signatures;

pub mod dyn_map;

pub mod item_tree;

pub mod lang_item;

pub mod hir;
pub use self::hir::type_ref;
pub mod expr_store;
pub mod resolver;

pub mod nameres;

pub mod src;

pub mod find_path;
pub mod import_map;
pub mod visibility;

use intern::{Interned, sym};
pub use rustc_abi as layout;
use triomphe::Arc;

pub use crate::signatures::LocalFieldId;

#[cfg(test)]
mod macro_expansion_tests;
#[cfg(test)]
mod test_db;

use std::hash::{Hash, Hasher};

use base_db::{Crate, impl_intern_key};
use hir_expand::{
    AstId, ExpandResult, ExpandTo, HirFileId, InFile, MacroCallId, MacroCallKind, MacroDefId,
    MacroDefKind,
    builtin::{BuiltinAttrExpander, BuiltinDeriveExpander, BuiltinFnLikeExpander, EagerExpander},
    db::ExpandDatabase,
    eager::expand_eager_macro_input,
    impl_intern_lookup,
    mod_path::ModPath,
    name::Name,
    proc_macro::{CustomProcMacroExpander, ProcMacroKind},
};
use item_tree::ExternBlock;
use la_arena::Idx;
use nameres::DefMap;
use span::{AstIdNode, Edition, FileAstId, SyntaxContext};
use stdx::impl_from;
use syntax::ast;

pub use hir_expand::{Intern, Lookup, tt};

use crate::{
    attr::Attrs,
    builtin_type::BuiltinType,
    db::DefDatabase,
    hir::generics::{LocalLifetimeParamId, LocalTypeOrConstParamId},
    item_tree::{
        Const, Enum, ExternCrate, Function, Impl, ItemTreeId, ItemTreeNode, Macro2, MacroRules,
        Static, Struct, Trait, TraitAlias, TypeAlias, Union, Use, Variant,
    },
    nameres::{LocalDefMap, block_def_map, crate_def_map, crate_local_def_map},
    signatures::VariantFields,
};

type FxIndexMap<K, V> = indexmap::IndexMap<K, V, rustc_hash::FxBuildHasher>;
/// A wrapper around three booleans
#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
pub struct ImportPathConfig {
    /// If true, prefer to unconditionally use imports of the `core` and `alloc` crate
    /// over the std.
    pub prefer_no_std: bool,
    /// If true, prefer import paths containing a prelude module.
    pub prefer_prelude: bool,
    /// If true, prefer abs path (starting with `::`) where it is available.
    pub prefer_absolute: bool,
    /// If true, paths containing `#[unstable]` segments may be returned, but only if if there is no
    /// stable path. This does not check, whether the item itself that is being imported is `#[unstable]`.
    pub allow_unstable: bool,
}

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
        impl_intern_key!($id, $loc);
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

type FunctionLoc = AssocItemLoc<Function>;
impl_intern!(FunctionId, FunctionLoc, intern_function, lookup_intern_function);
impl_loc!(FunctionLoc, id: Function, container: ItemContainerId);

type StructLoc = ItemLoc<Struct>;
impl_intern!(StructId, StructLoc, intern_struct, lookup_intern_struct);
impl_loc!(StructLoc, id: Struct, container: ModuleId);

pub type UnionLoc = ItemLoc<Union>;
impl_intern!(UnionId, UnionLoc, intern_union, lookup_intern_union);
impl_loc!(UnionLoc, id: Union, container: ModuleId);

pub type EnumLoc = ItemLoc<Enum>;
impl_intern!(EnumId, EnumLoc, intern_enum, lookup_intern_enum);
impl_loc!(EnumLoc, id: Enum, container: ModuleId);

type ConstLoc = AssocItemLoc<Const>;
impl_intern!(ConstId, ConstLoc, intern_const, lookup_intern_const);
impl_loc!(ConstLoc, id: Const, container: ItemContainerId);

pub type StaticLoc = AssocItemLoc<Static>;
impl_intern!(StaticId, StaticLoc, intern_static, lookup_intern_static);
impl_loc!(StaticLoc, id: Static, container: ItemContainerId);

pub type TraitLoc = ItemLoc<Trait>;
impl_intern!(TraitId, TraitLoc, intern_trait, lookup_intern_trait);
impl_loc!(TraitLoc, id: Trait, container: ModuleId);

pub type TraitAliasLoc = ItemLoc<TraitAlias>;
impl_intern!(TraitAliasId, TraitAliasLoc, intern_trait_alias, lookup_intern_trait_alias);
impl_loc!(TraitAliasLoc, id: TraitAlias, container: ModuleId);

type TypeAliasLoc = AssocItemLoc<TypeAlias>;
impl_intern!(TypeAliasId, TypeAliasLoc, intern_type_alias, lookup_intern_type_alias);
impl_loc!(TypeAliasLoc, id: TypeAlias, container: ItemContainerId);

type ImplLoc = ItemLoc<Impl>;
impl_intern!(ImplId, ImplLoc, intern_impl, lookup_intern_impl);
impl_loc!(ImplLoc, id: Impl, container: ModuleId);

type UseLoc = ItemLoc<Use>;
impl_intern!(UseId, UseLoc, intern_use, lookup_intern_use);
impl_loc!(UseLoc, id: Use, container: ModuleId);

type ExternCrateLoc = ItemLoc<ExternCrate>;
impl_intern!(ExternCrateId, ExternCrateLoc, intern_extern_crate, lookup_intern_extern_crate);
impl_loc!(ExternCrateLoc, id: ExternCrate, container: ModuleId);

type ExternBlockLoc = ItemLoc<ExternBlock>;
impl_intern!(ExternBlockId, ExternBlockLoc, intern_extern_block, lookup_intern_extern_block);
impl_loc!(ExternBlockLoc, id: ExternBlock, container: ModuleId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EnumVariantLoc {
    pub id: ItemTreeId<Variant>,
    pub parent: EnumId,
    pub index: u32,
}
impl_intern!(EnumVariantId, EnumVariantLoc, intern_enum_variant, lookup_intern_enum_variant);
impl_loc!(EnumVariantLoc, id: Variant, parent: EnumId);
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

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub struct BlockLoc {
    pub ast_id: AstId<ast::BlockExpr>,
    /// The containing module.
    pub module: ModuleId,
}
impl_intern!(BlockId, BlockLoc, intern_block, lookup_intern_block);

/// A `ModuleId` that is always a crate's root module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CrateRootModuleId {
    krate: Crate,
}

impl CrateRootModuleId {
    pub fn def_map(self, db: &dyn DefDatabase) -> &DefMap {
        crate_def_map(db, self.krate)
    }

    pub(crate) fn local_def_map(self, db: &dyn DefDatabase) -> (&DefMap, &LocalDefMap) {
        let def_map = crate_local_def_map(db, self.krate);
        (def_map.def_map(db), def_map.local(db))
    }

    pub fn krate(self) -> Crate {
        self.krate
    }
}

impl PartialEq<ModuleId> for CrateRootModuleId {
    fn eq(&self, other: &ModuleId) -> bool {
        other.block.is_none() && other.local_id == DefMap::ROOT && self.krate == other.krate
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

impl From<Crate> for CrateRootModuleId {
    fn from(krate: Crate) -> Self {
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
    krate: Crate,
    /// If this `ModuleId` was derived from a `DefMap` for a block expression, this stores the
    /// `BlockId` of that block expression. If `None`, this module is part of the crate-level
    /// `DefMap` of `krate`.
    block: Option<BlockId>,
    /// The module's ID in its originating `DefMap`.
    pub local_id: LocalModuleId,
}

impl ModuleId {
    pub fn def_map(self, db: &dyn DefDatabase) -> &DefMap {
        match self.block {
            Some(block) => block_def_map(db, block),
            None => crate_def_map(db, self.krate),
        }
    }

    pub(crate) fn local_def_map(self, db: &dyn DefDatabase) -> (&DefMap, &LocalDefMap) {
        match self.block {
            Some(block) => (block_def_map(db, block), self.only_local_def_map(db)),
            None => {
                let def_map = crate_local_def_map(db, self.krate);
                (def_map.def_map(db), def_map.local(db))
            }
        }
    }

    pub(crate) fn only_local_def_map(self, db: &dyn DefDatabase) -> &LocalDefMap {
        crate_local_def_map(db, self.krate).local(db)
    }

    pub fn crate_def_map(self, db: &dyn DefDatabase) -> &DefMap {
        crate_def_map(db, self.krate)
    }

    pub fn krate(self) -> Crate {
        self.krate
    }

    pub fn name(self, db: &dyn DefDatabase) -> Option<Name> {
        let def_map = self.def_map(db);
        let parent = def_map[self.local_id].parent?;
        def_map[parent].children.iter().find_map(|(name, module_id)| {
            if *module_id == self.local_id { Some(name.clone()) } else { None }
        })
    }

    /// Returns the module containing `self`, either the parent `mod`, or the module (or block) containing
    /// the block, if `self` corresponds to a block expression.
    pub fn containing_module(self, db: &dyn DefDatabase) -> Option<ModuleId> {
        self.def_map(db).containing_module(self.local_id)
    }

    pub fn containing_block(self) -> Option<BlockId> {
        self.block
    }

    pub fn is_block_module(self) -> bool {
        self.block.is_some() && self.local_id == DefMap::ROOT
    }

    pub fn is_within_block(self) -> bool {
        self.block.is_some()
    }

    /// Returns the [`CrateRootModuleId`] for this module if it is the crate root module.
    pub fn as_crate_root(&self) -> Option<CrateRootModuleId> {
        if self.local_id == DefMap::ROOT && self.block.is_none() {
            Some(CrateRootModuleId { krate: self.krate })
        } else {
            None
        }
    }

    /// Returns the [`CrateRootModuleId`] for this module.
    pub fn derive_crate_root(&self) -> CrateRootModuleId {
        CrateRootModuleId { krate: self.krate }
    }

    /// Whether this module represents the crate root module
    pub fn is_crate_root(&self) -> bool {
        self.local_id == DefMap::ROOT && self.block.is_none()
    }
}

/// An ID of a module, **local** to a `DefMap`.
pub type LocalModuleId = Idx<nameres::ModuleData>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FieldId {
    pub parent: VariantId,
    pub local_id: LocalFieldId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TupleId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TupleFieldId {
    pub tuple: TupleId,
    pub index: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LifetimeParamId {
    pub parent: GenericDefId,
    pub local_id: LocalLifetimeParamId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ItemContainerId {
    ExternBlockId(ExternBlockId),
    ModuleId(ModuleId),
    ImplId(ImplId),
    TraitId(TraitId),
}
impl_from!(ModuleId for ItemContainerId);

/// A Data Type
#[derive(Debug, PartialOrd, Ord, Clone, Copy, PartialEq, Eq, Hash, salsa_macros::Supertype)]
pub enum AdtId {
    StructId(StructId),
    UnionId(UnionId),
    EnumId(EnumId),
}
impl_from!(StructId, UnionId, EnumId for AdtId);

/// A macro
#[derive(Debug, PartialOrd, Ord, Clone, Copy, PartialEq, Eq, Hash, salsa_macros::Supertype)]
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

/// A constant, which might appears as a const item, an anonymous const block in expressions
/// or patterns, or as a constant in types with const generics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, salsa_macros::Supertype)]
pub enum GeneralConstId {
    ConstId(ConstId),
    StaticId(StaticId),
}

impl_from!(ConstId, StaticId for GeneralConstId);

impl GeneralConstId {
    pub fn generic_def(self, _db: &dyn DefDatabase) -> Option<GenericDefId> {
        match self {
            GeneralConstId::ConstId(it) => Some(it.into()),
            GeneralConstId::StaticId(it) => Some(it.into()),
        }
    }

    pub fn name(self, db: &dyn DefDatabase) -> String {
        match self {
            GeneralConstId::StaticId(it) => {
                let loc = it.lookup(db);
                let tree = loc.item_tree_id().item_tree(db);
                let name = tree[loc.id.value].name.display(db, Edition::CURRENT);
                name.to_string()
            }
            GeneralConstId::ConstId(const_id) => {
                let loc = const_id.lookup(db);
                let tree = loc.item_tree_id().item_tree(db);
                tree[loc.id.value].name.as_ref().map_or_else(
                    || "_".to_owned(),
                    |name| name.display(db, Edition::CURRENT).to_string(),
                )
            }
        }
    }
}

/// The defs which have a body (have root expressions for type inference).
#[derive(Debug, PartialOrd, Ord, Clone, Copy, PartialEq, Eq, Hash, salsa_macros::Supertype)]
pub enum DefWithBodyId {
    FunctionId(FunctionId),
    StaticId(StaticId),
    ConstId(ConstId),
    VariantId(EnumVariantId),
    // /// All fields of a variant are inference roots
    // VariantId(VariantId),
    // /// The signature can contain inference roots in a bunch of places
    // /// like const parameters or const arguments in paths
    // This should likely be kept on its own with a separate query
    // GenericDefId(GenericDefId),
}
impl_from!(FunctionId, ConstId, StaticId for DefWithBodyId);

impl From<EnumVariantId> for DefWithBodyId {
    fn from(id: EnumVariantId) -> Self {
        DefWithBodyId::VariantId(id)
    }
}

impl DefWithBodyId {
    pub fn as_generic_def_id(self, db: &dyn DefDatabase) -> Option<GenericDefId> {
        match self {
            DefWithBodyId::FunctionId(f) => Some(f.into()),
            DefWithBodyId::StaticId(s) => Some(s.into()),
            DefWithBodyId::ConstId(c) => Some(c.into()),
            DefWithBodyId::VariantId(c) => Some(c.lookup(db).parent.into()),
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

impl From<AssocItemId> for ModuleDefId {
    fn from(item: AssocItemId) -> Self {
        match item {
            AssocItemId::FunctionId(f) => f.into(),
            AssocItemId::ConstId(c) => c.into(),
            AssocItemId::TypeAliasId(t) => t.into(),
        }
    }
}

#[derive(Debug, PartialOrd, Ord, Clone, Copy, PartialEq, Eq, Hash, salsa_macros::Supertype)]
pub enum GenericDefId {
    AdtId(AdtId),
    // consts can have type parameters from their parents (i.e. associated consts of traits)
    ConstId(ConstId),
    FunctionId(FunctionId),
    ImplId(ImplId),
    // can't actually have generics currently, but they might in the future
    // More importantly, this completes the set of items that contain type references
    // which is to be used by the signature expression store in the future.
    StaticId(StaticId),
    TraitAliasId(TraitAliasId),
    TraitId(TraitId),
    TypeAliasId(TypeAliasId),
}
impl_from!(
    AdtId(StructId, EnumId, UnionId),
    ConstId,
    FunctionId,
    ImplId,
    StaticId,
    TraitAliasId,
    TraitId,
    TypeAliasId
    for GenericDefId
);

impl GenericDefId {
    pub fn file_id_and_params_of(
        self,
        db: &dyn DefDatabase,
    ) -> (HirFileId, Option<ast::GenericParamList>) {
        fn file_id_and_params_of_item_loc<Loc>(
            db: &dyn DefDatabase,
            def: impl Lookup<Database = dyn DefDatabase, Data = Loc>,
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
            GenericDefId::AdtId(AdtId::StructId(it)) => file_id_and_params_of_item_loc(db, it),
            GenericDefId::AdtId(AdtId::UnionId(it)) => file_id_and_params_of_item_loc(db, it),
            GenericDefId::AdtId(AdtId::EnumId(it)) => file_id_and_params_of_item_loc(db, it),
            GenericDefId::TraitId(it) => file_id_and_params_of_item_loc(db, it),
            GenericDefId::TraitAliasId(it) => file_id_and_params_of_item_loc(db, it),
            GenericDefId::ImplId(it) => file_id_and_params_of_item_loc(db, it),
            GenericDefId::ConstId(it) => (it.lookup(db).id.file_id(), None),
            GenericDefId::StaticId(it) => (it.lookup(db).id.file_id(), None),
        }
    }

    pub fn assoc_trait_container(self, db: &dyn DefDatabase) -> Option<TraitId> {
        match match self {
            GenericDefId::FunctionId(f) => f.lookup(db).container,
            GenericDefId::TypeAliasId(t) => t.lookup(db).container,
            GenericDefId::ConstId(c) => c.lookup(db).container,
            _ => return None,
        } {
            ItemContainerId::TraitId(trait_) => Some(trait_),
            _ => None,
        }
    }

    pub fn from_callable(db: &dyn DefDatabase, def: CallableDefId) -> GenericDefId {
        match def {
            CallableDefId::FunctionId(f) => f.into(),
            CallableDefId::StructId(s) => s.into(),
            CallableDefId::EnumVariantId(e) => e.lookup(db).parent.into(),
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

#[derive(Debug, PartialOrd, Ord, Clone, Copy, PartialEq, Eq, Hash, salsa_macros::Supertype)]
pub enum CallableDefId {
    FunctionId(FunctionId),
    StructId(StructId),
    EnumVariantId(EnumVariantId),
}

impl_from!(FunctionId, StructId, EnumVariantId for CallableDefId);
impl From<CallableDefId> for ModuleDefId {
    fn from(def: CallableDefId) -> ModuleDefId {
        match def {
            CallableDefId::FunctionId(f) => ModuleDefId::FunctionId(f),
            CallableDefId::StructId(s) => ModuleDefId::AdtId(AdtId::StructId(s)),
            CallableDefId::EnumVariantId(e) => ModuleDefId::EnumVariantId(e),
        }
    }
}

impl CallableDefId {
    pub fn krate(self, db: &dyn DefDatabase) -> Crate {
        match self {
            CallableDefId::FunctionId(f) => f.krate(db),
            CallableDefId::StructId(s) => s.krate(db),
            CallableDefId::EnumVariantId(e) => e.krate(db),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, salsa_macros::Supertype)]
pub enum VariantId {
    EnumVariantId(EnumVariantId),
    StructId(StructId),
    UnionId(UnionId),
}
impl_from!(EnumVariantId, StructId, UnionId for VariantId);

impl VariantId {
    pub fn variant_data(self, db: &dyn DefDatabase) -> Arc<VariantFields> {
        db.variant_fields(self)
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
    fn krate(&self, db: &dyn DefDatabase) -> Crate {
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
    ItemId: Lookup<Database = dyn DefDatabase, Data = ItemLoc<N>> + Copy,
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
    id: impl Lookup<Database = dyn DefDatabase, Data = AssocItemLoc<impl ItemTreeNode>>,
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

impl HasModule for DefWithBodyId {
    fn module(&self, db: &dyn DefDatabase) -> ModuleId {
        match self {
            DefWithBodyId::FunctionId(it) => it.module(db),
            DefWithBodyId::StaticId(it) => it.module(db),
            DefWithBodyId::ConstId(it) => it.module(db),
            DefWithBodyId::VariantId(it) => it.module(db),
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
            GenericDefId::ConstId(it) => it.module(db),
            GenericDefId::StaticId(it) => it.module(db),
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
/// Helper wrapper for `AstId` with `ModPath`
#[derive(Clone, Debug, Eq, PartialEq)]
struct AstIdWithPath<T: AstIdNode> {
    ast_id: AstId<T>,
    path: Interned<ModPath>,
}

impl<T: AstIdNode> AstIdWithPath<T> {
    fn new(file_id: HirFileId, ast_id: FileAstId<T>, path: Interned<ModPath>) -> AstIdWithPath<T> {
        AstIdWithPath { ast_id: AstId::new(file_id, ast_id), path }
    }
}

pub fn macro_call_as_call_id(
    db: &dyn ExpandDatabase,
    ast_id: AstId<ast::MacroCall>,
    path: &ModPath,
    call_site: SyntaxContext,
    expand_to: ExpandTo,
    krate: Crate,
    resolver: impl Fn(&ModPath) -> Option<MacroDefId> + Copy,
    eager_callback: &mut dyn FnMut(
        InFile<(syntax::AstPtr<ast::MacroCall>, span::FileAstId<ast::MacroCall>)>,
        MacroCallId,
    ),
) -> Result<ExpandResult<Option<MacroCallId>>, UnresolvedMacro> {
    let def = resolver(path).ok_or_else(|| UnresolvedMacro { path: path.clone() })?;

    let res = match def.kind {
        MacroDefKind::BuiltInEager(..) => expand_eager_macro_input(
            db,
            krate,
            &ast_id.to_node(db),
            ast_id,
            def,
            call_site,
            &|path| resolver(path).filter(MacroDefId::is_fn_like),
            eager_callback,
        ),
        _ if def.is_fn_like() => ExpandResult {
            value: Some(def.make_call(
                db,
                krate,
                MacroCallKind::FnLike { ast_id, expand_to, eager: None },
                call_site,
            )),
            err: None,
        },
        _ => return Err(UnresolvedMacro { path: path.clone() }),
    };
    Ok(res)
}

#[derive(Debug)]
pub struct UnresolvedMacro {
    pub path: ModPath,
}

#[derive(Default, Debug, Eq, PartialEq, Clone, Copy)]
pub struct SyntheticSyntax;

// Feature: Completions Attribute
// Crate authors can opt their type out of completions in some cases.
// This is done with the `#[rust_analyzer::completions(...)]` attribute.
//
// All completeable things support `#[rust_analyzer::completions(ignore_flyimport)]`,
// which causes the thing to get excluded from flyimport completion. It will still
// be completed when in scope. This is analogous to the setting `rust-analyzer.completion.autoimport.exclude`
// with `"type": "always"`.
//
// In addition, traits support two more modes: `#[rust_analyzer::completions(ignore_flyimport_methods)]`,
// which means the trait itself may still be flyimported but its methods won't, and
// `#[rust_analyzer::completions(ignore_methods)]`, which means the methods won't be completed even when
// the trait is in scope (but the trait itself may still be completed). The methods will still be completed
// on `dyn Trait`, `impl Trait` or where the trait is specified in bounds. These modes correspond to
// the settings `rust-analyzer.completion.autoimport.exclude` with `"type": "methods"` and
// `rust-analyzer.completion.excludeTraits`, respectively.
//
// Malformed attributes will be ignored without warnings.
//
// Note that users have no way to override this attribute, so be careful and only include things
// users definitely do not want to be completed!

/// `#[rust_analyzer::completions(...)]` options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Complete {
    /// No `#[rust_analyzer::completions(...)]`.
    Yes,
    /// `#[rust_analyzer::completions(ignore_flyimport)]`.
    IgnoreFlyimport,
    /// `#[rust_analyzer::completions(ignore_flyimport_methods)]` (on a trait only).
    IgnoreFlyimportMethods,
    /// `#[rust_analyzer::completions(ignore_methods)]` (on a trait only).
    IgnoreMethods,
}

impl Complete {
    pub fn extract(is_trait: bool, attrs: &Attrs) -> Complete {
        let mut do_not_complete = Complete::Yes;
        for ra_attr in attrs.rust_analyzer_tool() {
            let segments = ra_attr.path.segments();
            if segments.len() != 2 {
                continue;
            }
            let action = segments[1].symbol();
            if *action == sym::completions {
                match ra_attr.token_tree_value().map(|tt| tt.token_trees().flat_tokens()) {
                    Some([tt::TokenTree::Leaf(tt::Leaf::Ident(ident))]) => {
                        if ident.sym == sym::ignore_flyimport {
                            do_not_complete = Complete::IgnoreFlyimport;
                        } else if is_trait {
                            if ident.sym == sym::ignore_methods {
                                do_not_complete = Complete::IgnoreMethods;
                            } else if ident.sym == sym::ignore_flyimport_methods {
                                do_not_complete = Complete::IgnoreFlyimportMethods;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        do_not_complete
    }

    #[inline]
    pub fn for_trait_item(trait_attr: Complete, item_attr: Complete) -> Complete {
        match (trait_attr, item_attr) {
            (
                Complete::IgnoreFlyimportMethods
                | Complete::IgnoreFlyimport
                | Complete::IgnoreMethods,
                _,
            ) => Complete::IgnoreFlyimport,
            _ => item_attr,
        }
    }
}
