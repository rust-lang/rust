#![recursion_limit = "512"]

//! HIR (previously known as descriptors) provides a high-level object oriented
//! access to Rust code.
//!
//! The principal difference between HIR and syntax trees is that HIR is bound
//! to a particular crate instance. That is, it has cfg flags and features
//! applied. So, the relation between syntax and HIR is many-to-one.

macro_rules! impl_froms {
    ($e:ident: $($v:ident),*) => {
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
mod resolve;
pub mod diagnostics;

mod code_model;

#[cfg(test)]
mod marks;

use crate::{
    db::{AstDatabase, DefDatabase, HirDatabase, InternDatabase},
    ids::MacroFileKind,
    name::AsName,
    resolve::Resolver,
    source_id::{AstId, FileAstId},
};

pub use self::{
    adt::{AdtDef, VariantDef},
    either::Either,
    expr::ExprScopes,
    generics::{GenericParam, GenericParams, HasGenericParams},
    ids::{HirFileId, MacroCallId, MacroCallLoc, MacroDefId, MacroFile},
    impl_block::{ImplBlock, ImplItem},
    name::Name,
    nameres::{ImportId, Namespace, PerNs},
    path::{Path, PathKind},
    resolve::Resolution,
    source_binder::{PathResolution, ScopeEntryWithSyntax, SourceAnalyzer},
    source_id::{AstIdMap, ErasedFileAstId},
    ty::{display::HirDisplay, ApplicationTy, CallableDef, Substs, TraitRef, Ty, TypeCtor},
    type_ref::Mutability,
};

pub use self::code_model::{
    docs::{DocDef, Docs, Documentation},
    src::{HasSource, Source},
    BuiltinType, Const, ConstData, Container, Crate, CrateDependency, DefWithBody, Enum,
    EnumVariant, FieldSource, FnData, Function, MacroDef, Module, ModuleDef, ModuleSource, Static,
    Struct, StructField, Trait, TypeAlias, Union,
};
