//! This module implements some validity checks for attributes.
//! In particular it verifies that `#[inline]` and `#[repr]` attributes are
//! attached to items that actually support them and if there are
//! conflicts between multiple such attributes attached to the same
//! item.

use rustc_ast as ast;
use rustc_ast::visit as ast_visit;

use crate::hir;
use crate::{Item, ItemKind, TraitItem, TraitItemKind};

use crate::def::DefKind;
use std::fmt::{self, Display};

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum GenericParamKind {
    Type,
    Lifetime,
    Const,
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum MethodKind {
    Trait { body: bool },
    Inherent,
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Target {
    ExternCrate,
    Use,
    Static,
    Const,
    Fn,
    Closure,
    Mod,
    ForeignMod,
    GlobalAsm,
    TyAlias,
    OpaqueTy,
    ImplTraitPlaceholder,
    Enum,
    Variant,
    Struct,
    Field,
    Union,
    Trait,
    TraitAlias,
    Impl,
    Expression,
    Statement,
    Arm,
    AssocConst,
    Method(MethodKind),
    AssocTy,
    ForeignFn,
    ForeignStatic,
    ForeignTy,
    GenericParam(GenericParamKind),
    MacroDef,
    Param,
    PatField,
    ExprField,
}

impl Display for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", Self::name(*self))
    }
}

impl Target {
    pub fn from_item(item: &Item<'_>) -> Target {
        match item.kind {
            ItemKind::ExternCrate(..) => Target::ExternCrate,
            ItemKind::Use(..) => Target::Use,
            ItemKind::Static(..) => Target::Static,
            ItemKind::Const(..) => Target::Const,
            ItemKind::Fn(..) => Target::Fn,
            ItemKind::Macro(..) => Target::MacroDef,
            ItemKind::Mod(..) => Target::Mod,
            ItemKind::ForeignMod { .. } => Target::ForeignMod,
            ItemKind::GlobalAsm(..) => Target::GlobalAsm,
            ItemKind::TyAlias(..) => Target::TyAlias,
            ItemKind::OpaqueTy(ref opaque) => {
                if opaque.in_trait {
                    Target::ImplTraitPlaceholder
                } else {
                    Target::OpaqueTy
                }
            }
            ItemKind::Enum(..) => Target::Enum,
            ItemKind::Struct(..) => Target::Struct,
            ItemKind::Union(..) => Target::Union,
            ItemKind::Trait(..) => Target::Trait,
            ItemKind::TraitAlias(..) => Target::TraitAlias,
            ItemKind::Impl { .. } => Target::Impl,
        }
    }

    pub fn to_def_kind(self) -> DefKind {
        match self {
            Target::ExternCrate => DefKind::ExternCrate,
            Target::Use => DefKind::Use,
            Target::Static => DefKind::Static(ast::Mutability::Not),
            Target::Const => DefKind::Const,
            Target::Fn => DefKind::Fn,
            Target::Mod => DefKind::Mod,
            Target::ForeignMod => DefKind::ForeignMod,
            Target::GlobalAsm => DefKind::GlobalAsm,
            Target::TyAlias => DefKind::TyAlias,
            Target::OpaqueTy => DefKind::OpaqueTy,
            Target::ImplTraitPlaceholder => DefKind::ImplTraitPlaceholder,
            Target::Enum => DefKind::Enum,
            Target::Struct => DefKind::Struct,
            Target::Union => DefKind::Union,
            Target::Trait => DefKind::Trait,
            Target::TraitAlias => DefKind::TraitAlias,
            Target::Impl => DefKind::Impl,
            Target::Method(_) => DefKind::AssocFn,
            Target::Variant => DefKind::Variant,
            _ => panic!("unsupported Target::to_def_kind: {self:?}"),
        }
    }

    pub fn from_trait_item(trait_item: &TraitItem<'_>) -> Target {
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

    pub fn from_foreign_item(foreign_item: &hir::ForeignItem<'_>) -> Target {
        match foreign_item.kind {
            hir::ForeignItemKind::Fn(..) => Target::ForeignFn,
            hir::ForeignItemKind::Static(..) => Target::ForeignStatic,
            hir::ForeignItemKind::Type => Target::ForeignTy,
        }
    }

    pub fn from_generic_param(generic_param: &hir::GenericParam<'_>) -> Target {
        match generic_param.kind {
            hir::GenericParamKind::Type { .. } => Target::GenericParam(GenericParamKind::Type),
            hir::GenericParamKind::Lifetime { .. } => {
                Target::GenericParam(GenericParamKind::Lifetime)
            }
            hir::GenericParamKind::Const { .. } => Target::GenericParam(GenericParamKind::Const),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Target::ExternCrate => "extern crate",
            Target::Use => "use",
            Target::Static => "static item",
            Target::Const => "constant item",
            Target::Fn => "function",
            Target::Closure => "closure",
            Target::Mod => "module",
            Target::ForeignMod => "foreign module",
            Target::GlobalAsm => "global asm",
            Target::TyAlias => "type alias",
            Target::OpaqueTy => "opaque type",
            Target::ImplTraitPlaceholder => "opaque type in trait",
            Target::Enum => "enum",
            Target::Variant => "enum variant",
            Target::Struct => "struct",
            Target::Field => "struct field",
            Target::Union => "union",
            Target::Trait => "trait",
            Target::TraitAlias => "trait alias",
            Target::Impl => "implementation block",
            Target::Expression => "expression",
            Target::Statement => "statement",
            Target::Arm => "match arm",
            Target::AssocConst => "associated const",
            Target::Method(kind) => match kind {
                MethodKind::Inherent => "inherent method",
                MethodKind::Trait { body: false } => "required trait method",
                MethodKind::Trait { body: true } => "provided trait method",
            },
            Target::AssocTy => "associated type",
            Target::ForeignFn => "foreign function",
            Target::ForeignStatic => "foreign static item",
            Target::ForeignTy => "foreign type",
            Target::GenericParam(kind) => match kind {
                GenericParamKind::Type => "type parameter",
                GenericParamKind::Lifetime => "lifetime parameter",
                GenericParamKind::Const => "const parameter",
            },
            Target::MacroDef => "macro def",
            Target::Param => "function param",
            Target::PatField => "pattern field",
            Target::ExprField => "struct field",
        }
    }

    pub fn from_ast_item(item: &ast::Item) -> Target {
        match item.kind {
            ast::ItemKind::ExternCrate(_) => Target::ExternCrate,
            ast::ItemKind::Use(_) => Target::Use,
            ast::ItemKind::Static(..) => Target::Static,
            ast::ItemKind::Const(..) => Target::Const,
            ast::ItemKind::Fn(_) => Target::Fn,
            ast::ItemKind::Mod(..) => Target::Mod,
            ast::ItemKind::ForeignMod(_) => Target::ForeignMod,
            ast::ItemKind::GlobalAsm(_) => Target::GlobalAsm,
            ast::ItemKind::TyAlias(_) => Target::TyAlias,
            ast::ItemKind::Enum(..) => Target::Enum,
            ast::ItemKind::Struct(..) => Target::Struct,
            ast::ItemKind::Union(..) => Target::Union,
            ast::ItemKind::Trait(_) => Target::Trait,
            ast::ItemKind::TraitAlias(..) => Target::TraitAlias,
            ast::ItemKind::Impl(_) => Target::Impl,
            ast::ItemKind::MacroDef(_) => Target::MacroDef,
            ast::ItemKind::MacCall(_) => panic!("unexpected MacCall"),
        }
    }

    pub fn from_ast_assoc_item(kind: &ast::AssocItemKind, ctxt: ast_visit::AssocCtxt) -> Target {
        match kind {
            ast::AssocItemKind::Const(..) => Target::AssocConst,
            ast::AssocItemKind::Fn(f) => {
                let kind = match ctxt {
                    ast_visit::AssocCtxt::Impl => MethodKind::Inherent,
                    ast_visit::AssocCtxt::Trait => MethodKind::Trait { body: f.body.is_some() },
                };
                Target::Method(kind)
            }
            ast::AssocItemKind::Type(_) => Target::AssocTy,
            ast::AssocItemKind::MacCall(_) => panic!("unexpected MacCall"),
        }
    }
}
