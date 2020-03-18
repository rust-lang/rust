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
pub mod item_scope;

pub mod dyn_map;
pub mod keys;

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
pub mod child_by_source;

pub mod visibility;
pub mod find_path;

#[cfg(test)]
mod test_db;
#[cfg(test)]
mod marks;

use std::hash::Hash;

use hir_expand::{
    ast_id_map::FileAstId, eager::expand_eager_macro, hygiene::Hygiene, AstId, HirFileId, InFile,
    MacroCallId, MacroCallKind, MacroDefId, MacroDefKind,
};
use ra_arena::Idx;
use ra_db::{impl_intern_key, salsa, CrateId};
use ra_syntax::{ast, AstNode};

use crate::body::Expander;
use crate::builtin_type::BuiltinType;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModuleId {
    pub krate: CrateId,
    pub local_id: LocalModuleId,
}

/// An ID of a module, **local** to a specific crate
pub type LocalModuleId = Idx<nameres::ModuleData>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ItemLoc<N: AstNode> {
    pub container: ContainerId,
    pub ast_id: AstId<N>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AssocItemLoc<N: AstNode> {
    pub container: AssocContainerId,
    pub ast_id: AstId<N>,
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
type FunctionLoc = AssocItemLoc<ast::FnDef>;
impl_intern!(FunctionId, FunctionLoc, intern_function, lookup_intern_function);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StructId(salsa::InternId);
type StructLoc = ItemLoc<ast::StructDef>;
impl_intern!(StructId, StructLoc, intern_struct, lookup_intern_struct);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnionId(salsa::InternId);
pub type UnionLoc = ItemLoc<ast::UnionDef>;
impl_intern!(UnionId, UnionLoc, intern_union, lookup_intern_union);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EnumId(salsa::InternId);
pub type EnumLoc = ItemLoc<ast::EnumDef>;
impl_intern!(EnumId, EnumLoc, intern_enum, lookup_intern_enum);

// FIXME: rename to `VariantId`, only enums can ave variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EnumVariantId {
    pub parent: EnumId,
    pub local_id: LocalEnumVariantId,
}

pub type LocalEnumVariantId = Idx<adt::EnumVariantData>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StructFieldId {
    pub parent: VariantId,
    pub local_id: LocalStructFieldId,
}

pub type LocalStructFieldId = Idx<adt::StructFieldData>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConstId(salsa::InternId);
type ConstLoc = AssocItemLoc<ast::ConstDef>;
impl_intern!(ConstId, ConstLoc, intern_const, lookup_intern_const);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StaticId(salsa::InternId);
pub type StaticLoc = ItemLoc<ast::StaticDef>;
impl_intern!(StaticId, StaticLoc, intern_static, lookup_intern_static);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TraitId(salsa::InternId);
pub type TraitLoc = ItemLoc<ast::TraitDef>;
impl_intern!(TraitId, TraitLoc, intern_trait, lookup_intern_trait);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeAliasId(salsa::InternId);
type TypeAliasLoc = AssocItemLoc<ast::TypeAliasDef>;
impl_intern!(TypeAliasId, TypeAliasLoc, intern_type_alias, lookup_intern_type_alias);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImplId(salsa::InternId);
type ImplLoc = ItemLoc<ast::ImplDef>;
impl_intern!(ImplId, ImplLoc, intern_impl, lookup_intern_impl);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeParamId {
    pub parent: GenericDefId,
    pub local_id: LocalTypeParamId,
}

pub type LocalTypeParamId = Idx<generics::TypeParamData>;

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
    DefWithBodyId(DefWithBodyId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssocContainerId {
    ContainerId(ContainerId),
    ImplId(ImplId),
    TraitId(TraitId),
}
impl_froms!(AssocContainerId: ContainerId);

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
impl_froms!(VariantId: EnumVariantId, StructId, UnionId);

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

impl HasModule for ContainerId {
    fn module(&self, db: &dyn db::DefDatabase) -> ModuleId {
        match *self {
            ContainerId::ModuleId(it) => it,
            ContainerId::DefWithBodyId(it) => it.module(db),
        }
    }
}

impl HasModule for AssocContainerId {
    fn module(&self, db: &dyn db::DefDatabase) -> ModuleId {
        match *self {
            AssocContainerId::ContainerId(it) => it.module(db),
            AssocContainerId::ImplId(it) => it.lookup(db).container.module(db),
            AssocContainerId::TraitId(it) => it.lookup(db).container.module(db),
        }
    }
}

impl<N: AstNode> HasModule for AssocItemLoc<N> {
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
        .module(db)
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

impl HasModule for GenericDefId {
    fn module(&self, db: &dyn db::DefDatabase) -> ModuleId {
        match self {
            GenericDefId::FunctionId(it) => it.lookup(db).module(db),
            GenericDefId::AdtId(it) => it.module(db),
            GenericDefId::TraitId(it) => it.lookup(db).container.module(db),
            GenericDefId::TypeAliasId(it) => it.lookup(db).module(db),
            GenericDefId::ImplId(it) => it.lookup(db).container.module(db),
            GenericDefId::EnumVariantId(it) => it.parent.lookup(db).container.module(db),
            GenericDefId::ConstId(it) => it.lookup(db).module(db),
        }
    }
}

impl HasModule for StaticLoc {
    fn module(&self, db: &dyn db::DefDatabase) -> ModuleId {
        self.container.module(db)
    }
}

/// A helper trait for converting to MacroCallId
pub trait AsMacroCall {
    fn as_call_id(
        &self,
        db: &dyn db::DefDatabase,
        resolver: impl Fn(path::ModPath) -> Option<MacroDefId>,
    ) -> Option<MacroCallId>;
}

impl AsMacroCall for InFile<&ast::MacroCall> {
    fn as_call_id(
        &self,
        db: &dyn db::DefDatabase,
        resolver: impl Fn(path::ModPath) -> Option<MacroDefId>,
    ) -> Option<MacroCallId> {
        let ast_id = AstId::new(self.file_id, db.ast_id_map(self.file_id).ast_id(self.value));
        let h = Hygiene::new(db.upcast(), self.file_id);
        let path = path::ModPath::from_src(self.value.path()?, &h)?;

        AstIdWithPath::new(ast_id.file_id, ast_id.value, path).as_call_id(db, resolver)
    }
}

/// Helper wrapper for `AstId` with `ModPath`
#[derive(Clone, Debug, Eq, PartialEq)]
struct AstIdWithPath<T: ast::AstNode> {
    pub ast_id: AstId<T>,
    pub path: path::ModPath,
}

impl<T: ast::AstNode> AstIdWithPath<T> {
    pub fn new(file_id: HirFileId, ast_id: FileAstId<T>, path: path::ModPath) -> AstIdWithPath<T> {
        AstIdWithPath { ast_id: AstId::new(file_id, ast_id), path }
    }
}

impl AsMacroCall for AstIdWithPath<ast::MacroCall> {
    fn as_call_id(
        &self,
        db: &dyn db::DefDatabase,
        resolver: impl Fn(path::ModPath) -> Option<MacroDefId>,
    ) -> Option<MacroCallId> {
        let def: MacroDefId = resolver(self.path.clone())?;

        if let MacroDefKind::BuiltInEager(_) = def.kind {
            let macro_call = InFile::new(self.ast_id.file_id, self.ast_id.to_node(db.upcast()));
            let hygiene = Hygiene::new(db.upcast(), self.ast_id.file_id);

            Some(
                expand_eager_macro(db.upcast(), macro_call, def, &|path: ast::Path| {
                    resolver(path::ModPath::from_src(path, &hygiene)?)
                })?
                .into(),
            )
        } else {
            Some(def.as_lazy_macro(db.upcast(), MacroCallKind::FnLike(self.ast_id)).into())
        }
    }
}

impl AsMacroCall for AstIdWithPath<ast::ModuleItem> {
    fn as_call_id(
        &self,
        db: &dyn db::DefDatabase,
        resolver: impl Fn(path::ModPath) -> Option<MacroDefId>,
    ) -> Option<MacroCallId> {
        let def = resolver(self.path.clone())?;
        Some(
            def.as_lazy_macro(db.upcast(), MacroCallKind::Attr(self.ast_id, self.path.to_string()))
                .into(),
        )
    }
}
