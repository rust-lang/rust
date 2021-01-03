//! HIR (previously known as descriptors) provides a high-level object oriented
//! access to Rust code.
//!
//! The principal difference between HIR and syntax trees is that HIR is bound
//! to a particular crate instance. That is, it has cfg flags and features
//! applied. So, the relation between syntax and HIR is many-to-one.
//!
//! HIR is the public API of the all of the compiler logic above syntax trees.
//! It is written in "OO" style. Each type is self contained (as in, it knows it's
//! parents and full context). It should be "clean code".
//!
//! `hir_*` crates are the implementation of the compiler logic.
//! They are written in "ECS" style, with relatively little abstractions.
//! Many types are not self-contained, and explicitly use local indexes, arenas, etc.
//!
//! `hir` is what insulates the "we don't know how to actually write an incremental compiler"
//! from the ide with completions, hovers, etc. It is a (soft, internal) boundary:
//! https://www.tedinski.com/2018/02/06/system-boundaries.html.

#![recursion_limit = "512"]

mod semantics;
pub mod db;
mod source_analyzer;

pub mod diagnostics;

mod from_id;
mod code_model;
mod attrs;
mod has_source;

pub use crate::{
    attrs::{HasAttrs, Namespace},
    code_model::{
        Access, Adt, AsAssocItem, AssocItem, AssocItemContainer, Callable, CallableKind, Const,
        ConstParam, Crate, CrateDependency, DefWithBody, Enum, Field, FieldSource, Function,
        GenericDef, GenericParam, HasVisibility, Impl, Label, LifetimeParam, Local, MacroDef,
        Module, ModuleDef, ScopeDef, Static, Struct, Trait, Type, TypeAlias, TypeParam, Union,
        Variant, VariantDef,
    },
    has_source::HasSource,
    semantics::{PathResolution, Semantics, SemanticsScope},
};

pub use hir_def::{
    adt::StructKind,
    attr::{Attrs, Documentation},
    body::scope::ExprScopes,
    builtin_type::BuiltinType,
    find_path::PrefixKind,
    import_map,
    item_scope::ItemInNs,
    nameres::ModuleSource,
    path::{ModPath, PathKind},
    type_ref::{Mutability, TypeRef},
    visibility::Visibility,
};
pub use hir_expand::{
    name::{known, AsName, Name},
    ExpandResult, HirFileId, InFile, MacroCallId, MacroCallLoc, /* FIXME */ MacroDefId,
    MacroFile, Origin,
};
pub use hir_ty::display::HirDisplay;

// These are negative re-exports: pub using these names is forbidden, they
// should remain private to hir internals.
#[allow(unused)]
use {hir_def::path::Path, hir_expand::hygiene::Hygiene};
