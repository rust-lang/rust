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
//! `ra_hir_*` crates are the implementation of the compiler logic.
//! They are written in "ECS" style, with relatively little abstractions.
//! Many types are not self-contained, and explicitly use local indexes, arenas, etc.
//!
//! `ra_hir` is what insulates the "we don't know how to actually write an incremental compiler"
//! from the ide with completions, hovers, etc. It is a (soft, internal) boundary:
//! https://www.tedinski.com/2018/02/06/system-boundaries.html.

#![recursion_limit = "512"]

macro_rules! impl_froms {
    ($e:ident: $($v:ident $(($($sv:ident),*))?),*$(,)?) => {
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

mod semantics;
pub mod db;
mod source_analyzer;

pub mod diagnostics;

mod from_id;
mod code_model;

mod has_source;

pub use crate::{
    code_model::{
        Adt, AsAssocItem, AssocItem, AssocItemContainer, AttrDef, Const, Crate, CrateDependency,
        DefWithBody, Docs, Enum, EnumVariant, Field, FieldSource, Function, GenericDef, HasAttrs,
        HasVisibility, ImplDef, Local, MacroDef, Module, ModuleDef, ScopeDef, Static, Struct,
        Trait, Type, TypeAlias, TypeParam, Union, VariantDef, Visibility,
    },
    has_source::HasSource,
    semantics::{original_range, PathResolution, Semantics, SemanticsScope},
};

pub use hir_def::{
    adt::StructKind,
    attr::Attrs,
    body::scope::ExprScopes,
    builtin_type::BuiltinType,
    docs::Documentation,
    nameres::ModuleSource,
    path::{ModPath, Path, PathKind},
    type_ref::Mutability,
};
pub use hir_expand::{
    hygiene::Hygiene, name::Name, HirFileId, InFile, MacroCallId, MacroCallLoc, MacroDefId,
    MacroFile, Origin,
};
pub use hir_ty::{display::HirDisplay, CallableDef};
