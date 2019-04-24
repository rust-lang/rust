//! HIR (previously known as descriptors) provides a high-level object oriented
//! access to Rust code.
//!
//! The principal difference between HIR and syntax trees is that HIR is bound
//! to a particular crate instance. That is, it has cfg flags and features
//! applied. So, the relation between syntax and HIR is many-to-one.

macro_rules! impl_froms {
    ($e:ident: $($v:ident), *) => {
        $(
            impl From<$v> for $e {
                fn from(it: $v) -> $e {
                    $e::$v(it)
                }
            }
        )*
    }
}

mod either;

pub mod db;
#[macro_use]
pub mod mock;
mod path;
pub mod source_binder;

mod source_id;
mod ids;
mod name;
mod nameres;
mod adt;
mod traits;
mod type_alias;
mod type_ref;
mod ty;
mod impl_block;
mod expr;
mod lang_item;
mod generics;
mod docs;
mod resolve;
pub mod diagnostics;

mod code_model_api;
mod code_model_impl;

#[cfg(test)]
mod marks;

use crate::{
    db::{HirDatabase, DefDatabase},
    name::{AsName, KnownName},
    source_id::{FileAstId, AstId},
    resolve::Resolver,
};

pub use self::{
    either::Either,
    path::{Path, PathKind},
    name::Name,
    source_id::{AstIdMap, ErasedFileAstId},
    ids::{HirFileId, MacroDefId, MacroCallId, MacroCallLoc},
    nameres::{PerNs, Namespace, ImportId},
    ty::{Ty, ApplicationTy, TypeCtor, TraitRef, Substs, display::HirDisplay, CallableDef},
    impl_block::{ImplBlock, ImplItem},
    docs::{Docs, Documentation},
    adt::AdtDef,
    expr::ExprScopes,
    resolve::Resolution,
    generics::{GenericParams, GenericParam, HasGenericParams},
    source_binder::{SourceAnalyzer, PathResolution, ScopeEntryWithSyntax,MacroByExampleDef},
};

pub use self::code_model_api::{
    Crate, CrateDependency,
    DefWithBody,
    Module, ModuleDef, ModuleSource,
    Struct, Enum, EnumVariant,
    Function, FnSignature,
    StructField, FieldSource,
    Static, Const, ConstSignature,
    Trait, TypeAlias, Container
};
