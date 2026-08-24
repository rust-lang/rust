//! Implements conversions from HIR types to Target.

use rustc_attr_ir::target::{GenericParamKind, MethodKind, Target};

use crate::def::DefKind;
use crate::{self as hir, ItemKind, TraitItemKind};

impl From<&hir::ForeignItem<'_>> for Target {
    fn from(foreign_item: &hir::ForeignItem<'_>) -> Target {
        match foreign_item.kind {
            hir::ForeignItemKind::Fn(..) => Target::ForeignFn,
            hir::ForeignItemKind::Static(..) => Target::ForeignStatic,
            hir::ForeignItemKind::Type => Target::ForeignTy,
        }
    }
}

impl From<&hir::GenericParam<'_>> for Target {
    fn from(generic_param: &hir::GenericParam<'_>) -> Target {
        match generic_param.kind {
            hir::GenericParamKind::Type { default, .. } => Target::GenericParam {
                kind: GenericParamKind::Type,
                has_default: default.is_some(),
            },
            hir::GenericParamKind::Lifetime { .. } => {
                Target::GenericParam { kind: GenericParamKind::Lifetime, has_default: false }
            }
            hir::GenericParamKind::Const { default, .. } => Target::GenericParam {
                kind: GenericParamKind::Const,
                has_default: default.is_some(),
            },
        }
    }
}

impl From<&hir::TraitItem<'_>> for Target {
    fn from(trait_item: &hir::TraitItem<'_>) -> Target {
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

impl From<DefKind> for Target {
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

impl From<&hir::Item<'_>> for Target {
    fn from(item: &hir::Item<'_>) -> Target {
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
