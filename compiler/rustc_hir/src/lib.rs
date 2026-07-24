//! HIR datatypes. See the [rustc dev guide] for more info.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/hir.html

// tidy-alphabetical-start
#![feature(associated_type_defaults)]
#![feature(closure_track_caller)]
#![feature(const_default)]
#![feature(const_trait_impl)]
#![feature(default_field_values)]
#![feature(derive_const)]
#![feature(exhaustive_patterns)]
#![feature(never_type)]
#![recursion_limit = "256"]
// tidy-alphabetical-end

extern crate self as rustc_hir;

mod arena;

pub mod def;
pub mod def_path_hash_map;
pub mod definitions;

pub use rustc_span::def_id;
mod hir;
pub use rustc_hir_id::{self as hir_id, *};
pub mod intravisit;
pub mod attrs {
    pub use rustc_attr_ir::*;
}
pub use rustc_attr_ir::*;
pub mod lints;
pub mod pat_util;

mod stable_hash_impls;
#[cfg(test)]
mod tests;

#[doc(no_inline)]
pub use hir::*;
pub use rustc_ast::attr::version::*;

use crate::def::DefKind;
arena_types!(rustc_arena::declare_arena);

impl From<&ForeignItem<'_>> for Target {
    fn from(foreign_item: &ForeignItem<'_>) -> Target {
        match foreign_item.kind {
            hir::ForeignItemKind::Fn(..) => Target::ForeignFn,
            hir::ForeignItemKind::Static(..) => Target::ForeignStatic,
            hir::ForeignItemKind::Type => Target::ForeignTy,
        }
    }
}
impl From<&GenericParam<'_>> for Target {
    fn from(generic_param: &GenericParam<'_>) -> Target {
        match generic_param.kind {
            GenericParamKind::Type { default, .. } => Target::GenericParam {
                kind: rustc_attr_ir::target::GenericParamKind::Type,
                has_default: default.is_some(),
            },
            hir::GenericParamKind::Lifetime { .. } => Target::GenericParam {
                kind: rustc_attr_ir::target::GenericParamKind::Lifetime,
                has_default: false,
            },
            hir::GenericParamKind::Const { default, .. } => Target::GenericParam {
                kind: rustc_attr_ir::target::GenericParamKind::Const,
                has_default: default.is_some(),
            },
        }
    }
}
impl From<DefKind> for Target {
    // FIXME: For now, should only be used with def_kinds from ItemIds
    fn from(def_kind: DefKind) -> Target {
        match def_kind {
            DefKind::ExternCrate => Target::ExternCrate,
            DefKind::Use => Target::Use,
            DefKind::Static { .. } => Target::Static,
            DefKind::Const { .. } => Target::Const,
            DefKind::Fn => Target::Fn,
            DefKind::Macro(..) => Target::MacroDef,
            DefKind::Mod => Target::Mod,
            DefKind::ForeignMod => Target::ForeignMod,
            DefKind::GlobalAsm => Target::GlobalAsm,
            DefKind::TyAlias => Target::TyAlias,
            DefKind::Enum => Target::Enum,
            DefKind::Struct => Target::Struct,
            DefKind::Union => Target::Union,
            DefKind::Trait => Target::Trait,
            DefKind::TraitAlias => Target::TraitAlias,
            DefKind::Impl { of_trait } => Target::Impl { of_trait },
            _ => panic!("impossible case reached"),
        }
    }
}
impl From<&TraitItem<'_>> for Target {
    fn from(trait_item: &TraitItem<'_>) -> Target {
        match trait_item.kind {
            TraitItemKind::Const(..) => Target::AssocConst,
            TraitItemKind::Fn(_, hir::TraitFn::Required(_)) => {
                Target::Method(MethodKind::Trait { body: false })
            }
            TraitItemKind::Fn(_, hir::TraitFn::Provided(_)) => {
                Target::Method(MethodKind::Trait { body: true })
            }
            TraitItemKind::Type(..) => Target::AssocTy,
        }
    }
}
impl From<&Item<'_>> for Target {
    fn from(item: &Item<'_>) -> Target {
        match item.kind {
            ItemKind::ExternCrate(..) => Target::ExternCrate,
            ItemKind::Use(..) => Target::Use,
            ItemKind::Static { .. } => Target::Static,
            ItemKind::Const(..) => Target::Const,
            ItemKind::Fn { .. } => Target::Fn,
            ItemKind::Macro(..) => Target::MacroDef,
            ItemKind::Mod(..) => Target::Mod,
            ItemKind::ForeignMod { .. } => Target::ForeignMod,
            ItemKind::GlobalAsm { .. } => Target::GlobalAsm,
            ItemKind::TyAlias(..) => Target::TyAlias,
            ItemKind::Enum(..) => Target::Enum,
            ItemKind::Struct(..) => Target::Struct,
            ItemKind::Union(..) => Target::Union,
            ItemKind::Trait { .. } => Target::Trait,
            ItemKind::TraitAlias(..) => Target::TraitAlias,
            ItemKind::Impl(imp_) => Target::Impl { of_trait: imp_.of_trait.is_some() },
        }
    }
}
