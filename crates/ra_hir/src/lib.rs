//! HIR (previously known as descriptors) provides a high-level object oriented
//! access to Rust code.
//!
//! The principal difference between HIR and syntax trees is that HIR is bound
//! to a particular crate instance. That is, it has cfg flags and features
//! applied. So, the relation between syntax and HIR is many-to-one.

#![recursion_limit = "512"]

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

pub mod db;
pub mod source_binder;

pub mod diagnostics;

mod from_id;
mod code_model;

mod has_source;
mod from_source;

pub use crate::{
    code_model::{
        Adt, AssocItem, AttrDef, Const, Crate, CrateDependency, DefWithBody, Docs, Enum,
        EnumVariant, FieldSource, Function, GenericDef, HasAttrs, ImplBlock, Local, MacroDef,
        Module, ModuleDef, ScopeDef, Static, Struct, StructField, Trait, Type, TypeAlias,
        TypeParam, Union, VariantDef,
    },
    from_source::FromSource,
    has_source::HasSource,
    source_binder::{PathResolution, ScopeEntryWithSyntax, SourceAnalyzer},
};

pub use hir_def::{
    body::scope::ExprScopes,
    builtin_type::BuiltinType,
    docs::Documentation,
    nameres::ModuleSource,
    path::{ModPath, Path, PathKind},
    type_ref::Mutability,
};
pub use hir_expand::{
    name::Name, HirFileId, InFile, MacroCallId, MacroCallLoc, MacroDefId, MacroFile, Origin,
};
pub use hir_ty::{display::HirDisplay, CallableDef};
