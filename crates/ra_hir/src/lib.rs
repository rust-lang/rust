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

mod ids;
mod adt;
mod traits;
mod type_alias;
mod ty;
mod impl_block;
mod expr;
mod lang_item;
pub mod generics;
mod resolve;
pub mod diagnostics;
mod util;

mod from_id;
mod code_model;

pub mod from_source;

#[cfg(test)]
mod test_db;
#[cfg(test)]
mod marks;

use crate::resolve::Resolver;

pub use crate::{
    adt::VariantDef,
    code_model::ImplBlock,
    code_model::{
        attrs::{AttrDef, Attrs},
        docs::{DocDef, Docs, Documentation},
        src::{HasBodySource, HasSource},
        Adt, AssocItem, Const, ConstData, Container, Crate, CrateDependency, DefWithBody, Enum,
        EnumVariant, FieldSource, FnData, Function, GenericParam, HasBody, Local, MacroDef, Module,
        ModuleDef, ModuleSource, Static, Struct, StructField, Trait, TypeAlias, Union,
    },
    expr::ExprScopes,
    from_source::FromSource,
    generics::GenericDef,
    ids::{HirFileId, MacroCallId, MacroCallLoc, MacroDefId, MacroFile},
    resolve::ScopeDef,
    source_binder::{PathResolution, ScopeEntryWithSyntax, SourceAnalyzer},
    ty::{
        display::HirDisplay,
        primitive::{FloatBitness, FloatTy, IntBitness, IntTy, Signedness, Uncertain},
        ApplicationTy, CallableDef, Substs, TraitRef, Ty, TypeCtor, TypeWalk,
    },
};

pub use hir_def::{
    builtin_type::BuiltinType,
    nameres::{per_ns::PerNs, raw::ImportId},
    path::{Path, PathKind},
    type_ref::Mutability,
};
pub use hir_expand::{either::Either, name::Name, Source};
