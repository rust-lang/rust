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

use std::{
    hash::{Hash, Hasher},
    panic::{RefUnwindSafe, UnwindSafe},
};

use base_db::{impl_intern_key, salsa, CrateId, ProcMacroKind};
use hir_expand::{
    ast_id_map::FileAstId,
    attrs::{Attr, AttrId, AttrInput},
    builtin_attr_macro::BuiltinAttrExpander,
    builtin_derive_macro::BuiltinDeriveExpander,
    builtin_fn_macro::{BuiltinFnLikeExpander, EagerExpander},
    db::ExpandDatabase,
    eager::expand_eager_macro_input,
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
        Const, Enum, Function, Impl, ItemTreeId, ItemTreeNode, MacroDef, MacroRules, Static,
        Struct, Trait, TraitAlias, TypeAlias, Union,
    },
};

/// A `ModuleId` that is always a crate's root module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CrateRootModuleId {
    krate: CrateId,
}

impl CrateRootModuleId {
    pub fn def_map(&self, db: &dyn db::DefDatabase) -> Arc<DefMap> {
        db.crate_def_map(self.krate)
    }

    pub fn krate(self) -> CrateId {
        self.krate
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
    pub container: CrateRootModuleId,
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
    // FIXME(const-generic-body): ModuleId should not be a type owner. This needs to be fixed to make `TypeOwnerId` actually
    // useful for assigning ids to in type consts.
    ModuleId(ModuleId),
}

impl TypeOwnerId {
    fn as_generic_def_id(self) -> Option<GenericDefId> {
        Some(match self {
            TypeOwnerId::FunctionId(x) => GenericDefId::FunctionId(x),
            TypeOwnerId::ConstId(x) => GenericDefId::ConstId(x),
            TypeOwnerId::AdtId(x) => GenericDefId::AdtId(x),
            TypeOwnerId::TraitId(x) => GenericDefId::TraitId(x),
            TypeOwnerId::TraitAliasId(x) => GenericDefId::TraitAliasId(x),
            TypeOwnerId::TypeAliasId(x) => GenericDefId::TypeAliasId(x),
            TypeOwnerId::ImplId(x) => GenericDefId::ImplId(x),
            TypeOwnerId::EnumVariantId(x) => GenericDefId::EnumVariantId(x),
            TypeOwnerId::InTypeConstId(_) | TypeOwnerId::ModuleId(_) | TypeOwnerId::StaticId(_) => {
                return None
            }
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
    EnumVariantId,
    ModuleId
    for TypeOwnerId
);

// Every `DefWithBodyId` is a type owner, since bodies can contain type (e.g. `{ let x: Type = _; }`)
impl From<DefWithBodyId> for TypeOwnerId {
    fn from(value: DefWithBodyId) -> Self {
        match value {
            DefWithBodyId::FunctionId(x) => x.into(),
            DefWithBodyId::StaticId(x) => x.into(),
            DefWithBodyId::ConstId(x) => x.into(),
            DefWithBodyId::InTypeConstId(x) => x.into(),
            DefWithBodyId::VariantId(x) => x.into(),
        }
    }
}

impl From<GenericDefId> for TypeOwnerId {
    fn from(value: GenericDefId) -> Self {
        match value {
            GenericDefId::FunctionId(x) => x.into(),
            GenericDefId::AdtId(x) => x.into(),
            GenericDefId::TraitId(x) => x.into(),
            GenericDefId::TraitAliasId(x) => x.into(),
            GenericDefId::TypeAliasId(x) => x.into(),
            GenericDefId::ImplId(x) => x.into(),
            GenericDefId::EnumVariantId(x) => x.into(),
            GenericDefId::ConstId(x) => x.into(),
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

#[derive(Debug, Hash, Eq, Clone)]
pub struct InTypeConstLoc {
    pub id: AstId<ast::ConstArg>,
    /// The thing this const arg appears in
    pub owner: TypeOwnerId,
    pub thing: Box<dyn OpaqueInternableThing>,
}

impl PartialEq for InTypeConstLoc {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.owner == other.owner && &*self.thing == &*other.thing
    }
}

impl InTypeConstId {
    pub fn source(&self, db: &dyn db::DefDatabase) -> ast::ConstArg {
        let src = self.lookup(db).id;
        let file_id = src.file_id;
        let root = &db.parse_or_expand(file_id);
        db.ast_id_map(file_id).get(src.value).to_node(root)
    }
}

/// A constant, which might appears as a const item, an annonymous const block in expressions
/// or patterns, or as a constant in types with const generics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GeneralConstId {
    ConstId(ConstId),
    ConstBlockId(ConstBlockId),
    InTypeConstId(InTypeConstId),
}

impl_from!(ConstId, ConstBlockId, InTypeConstId for GeneralConstId);

impl GeneralConstId {
    pub fn generic_def(self, db: &dyn db::DefDatabase) -> Option<GenericDefId> {
        match self {
            GeneralConstId::ConstId(it) => Some(it.into()),
            GeneralConstId::ConstBlockId(it) => it.lookup(db).parent.as_generic_def_id(),
            GeneralConstId::InTypeConstId(it) => it.lookup(db).owner.as_generic_def_id(),
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
            MacroId::ProcMacroId(it) => it.lookup(db).container.into(),
        }
    }
}

impl HasModule for TypeOwnerId {
    fn module(&self, db: &dyn db::DefDatabase) -> ModuleId {
        match self {
            TypeOwnerId::FunctionId(x) => x.lookup(db).module(db),
            TypeOwnerId::StaticId(x) => x.lookup(db).module(db),
            TypeOwnerId::ConstId(x) => x.lookup(db).module(db),
            TypeOwnerId::InTypeConstId(x) => x.lookup(db).owner.module(db),
            TypeOwnerId::AdtId(x) => x.module(db),
            TypeOwnerId::TraitId(x) => x.lookup(db).container,
            TypeOwnerId::TraitAliasId(x) => x.lookup(db).container,
            TypeOwnerId::TypeAliasId(x) => x.lookup(db).module(db),
            TypeOwnerId::ImplId(x) => x.lookup(db).container,
            TypeOwnerId::EnumVariantId(x) => x.parent.lookup(db).container,
            TypeOwnerId::ModuleId(x) => *x,
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
            DefWithBodyId::InTypeConstId(it) => it.lookup(db).owner.module(db),
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
            return Ok(ExpandResult::only_err(ExpandError::other("malformed macro invocation")));
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
        expand_eager_macro_input(db, krate, macro_call, def, &resolver)?
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
        Some(AttrInput::TokenTree(tt)) => (
            {
                let mut tt = tt.0.clone();
                tt.delimiter = tt::Delimiter::UNSPECIFIED;
                tt
            },
            tt.1.clone(),
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
