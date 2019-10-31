//! hir makes heavy use of ids: integer (u32) handlers to various things. You
//! can think of id as a pointer (but without a lifetime) or a file descriptor
//! (but for hir objects).
//!
//! This module defines a bunch of ids we are using. The most important ones are
//! probably `HirFileId` and `DefId`.

use ra_db::salsa;

pub use hir_def::{
    AstItemDef, ConstId, EnumId, FunctionId, ItemLoc, LocationCtx, StaticId, StructId, TraitId,
    TypeAliasId,
};
pub use hir_expand::{HirFileId, MacroCallId, MacroCallLoc, MacroDefId, MacroFile, MacroFileKind};

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

/// This exists just for Chalk, because Chalk just has a single `StructId` where
/// we have different kinds of ADTs, primitive types and special type
/// constructors like tuples and function pointers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeCtorId(salsa::InternId);
impl_intern_key!(TypeCtorId);

/// This exists just for Chalk, because our ImplIds are only unique per module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlobalImplId(salsa::InternId);
impl_intern_key!(GlobalImplId);
