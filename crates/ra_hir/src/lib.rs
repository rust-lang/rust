//! HIR (previsouly known as descriptors) provides a high-level OO acess to Rust
//! code.
//!
//! The principal difference between HIR and syntax trees is that HIR is bound
//! to a particular crate instance. That is, it has cfg flags and features
//! applied. So, there relation between syntax and HIR is many-to-one.

macro_rules! ctry {
    ($expr:expr) => {
        match $expr {
            None => return Ok(None),
            Some(it) => it,
        }
    };
}

pub mod db;
#[cfg(test)]
mod mock;
mod query_definitions;
mod path;
pub mod source_binder;

mod ids;
mod macros;
mod name;
mod module_tree;
mod nameres;
mod function;
mod adt;
mod type_ref;
mod ty;
mod impl_block;
mod expr;

mod code_model_api;
mod code_model_impl;

use crate::{
    db::HirDatabase,
    name::{AsName, KnownName},
    ids::{DefKind, SourceItemId, SourceFileItemId, SourceFileItems},
};

pub use self::{
    path::{Path, PathKind},
    name::Name,
    ids::{HirFileId, DefId, DefLoc, MacroCallId, MacroCallLoc},
    macros::{MacroDef, MacroInput, MacroExpansion},
    module_tree::ModuleId,
    nameres::{ItemMap, PerNs, Namespace, Resolution},
    function::{Function, FnSignature, FnScopes, ScopesWithSyntaxMapping},
    adt::{Struct, Enum},
    ty::Ty,
    impl_block::{ImplBlock, ImplItem},
};

pub use self::function::FnSignatureInfo;

pub use self::code_model_api::{
    Crate, CrateDependency,
    Module, ModuleSource, Problem,
};

pub enum Def {
    Module(Module),
    Function(Function),
    Struct(Struct),
    Enum(Enum),
    Item,
}
