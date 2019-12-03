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

pub mod debug;

pub mod db;
pub mod source_binder;

mod ty;
pub mod diagnostics;

mod from_id;
mod code_model;

pub mod from_source;

pub use crate::{
    code_model::{
        src::HasSource, Adt, AssocItem, AttrDef, Const, Container, Crate, CrateDependency,
        DefWithBody, Docs, Enum, EnumVariant, FieldSource, Function, GenericDef, GenericParam,
        HasAttrs, ImplBlock, Import, Local, MacroDef, Module, ModuleDef, ModuleSource, ScopeDef,
        Static, Struct, StructField, Trait, Type, TypeAlias, Union, VariantDef,
    },
    from_source::FromSource,
    source_binder::{PathResolution, ScopeEntryWithSyntax, SourceAnalyzer},
    ty::{
        display::HirDisplay,
        primitive::{FloatBitness, FloatTy, IntBitness, IntTy, Signedness, Uncertain},
        ApplicationTy, CallableDef, Substs, TraitRef, Ty, TypeCtor, TypeWalk,
    },
};

pub use hir_def::{
    body::scope::ExprScopes,
    builtin_type::BuiltinType,
    docs::Documentation,
    path::{Path, PathKind},
    type_ref::Mutability,
};
pub use hir_expand::{
    name::Name, HirFileId, InFile, MacroCallId, MacroCallLoc, MacroDefId, MacroFile,
};
