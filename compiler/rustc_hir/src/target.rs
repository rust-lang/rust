//! This module implements some validity checks for attributes.
//! In particular it verifies that `#[inline]` and `#[repr]` attributes are
//! attached to items that actually support them and if there are
//! conflicts between multiple such attributes attached to the same
//! item.

use std::fmt::{self, Display};

use rustc_ast::visit::AssocCtxt;
use rustc_ast::{AssocItemKind, ForeignItemKind, ast};
use rustc_macros::HashStable_Generic;

use crate::def::DefKind;
use crate::{Item, ItemKind, TraitItem, TraitItemKind, hir};

#[derive(Copy, Clone, PartialEq, Debug, Eq, HashStable_Generic)]
pub enum GenericParamKind {
    Type,
    Lifetime,
    Const,
}

#[derive(Copy, Clone, PartialEq, Debug, Eq, HashStable_Generic)]
pub enum MethodKind {
    /// Method in a `trait Trait` block
    Trait {
        /// Whether a default is provided for this method
        body: bool,
    },
    /// Method in a `impl Trait for Type` block
    TraitImpl,
    /// Method in a `impl Type` block
    Inherent,
}

#[derive(Copy, Clone, PartialEq, Debug, Eq, HashStable_Generic)]
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
    Enum,
    Variant,
    Struct,
    Field,
    Union,
    Trait,
    TraitAlias,
    Impl { of_trait: bool },
    Expression,
    Statement,
    Arm,
    AssocConst,
    Method(MethodKind),
    AssocTy,
    ForeignFn,
    ForeignStatic,
    ForeignTy,
    GenericParam { kind: GenericParamKind, has_default: bool },
    MacroDef,
    Param,
    PatField,
    ExprField,
    WherePredicate,
    MacroCall,
    Crate,
    Delegation { mac: bool },
}

impl Display for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", Self::name(*self))
    }
}

rustc_error_messages::into_diag_arg_using_display!(Target);

impl Target {
    pub fn is_associated_item(self) -> bool {
        match self {
            Target::AssocConst | Target::AssocTy | Target::Method(_) => true,
            Target::ExternCrate
            | Target::Use
            | Target::Static
            | Target::Const
            | Target::Fn
            | Target::Closure
            | Target::Mod
            | Target::ForeignMod
            | Target::GlobalAsm
            | Target::TyAlias
            | Target::Enum
            | Target::Variant
            | Target::Struct
            | Target::Field
            | Target::Union
            | Target::Trait
            | Target::TraitAlias
            | Target::Impl { .. }
            | Target::Expression
            | Target::Statement
            | Target::Arm
            | Target::ForeignFn
            | Target::ForeignStatic
            | Target::ForeignTy
            | Target::GenericParam { .. }
            | Target::MacroDef
            | Target::Param
            | Target::PatField
            | Target::ExprField
            | Target::MacroCall
            | Target::Crate
            | Target::WherePredicate
            | Target::Delegation { .. } => false,
        }
    }

    pub fn from_item(item: &Item<'_>) -> Target {
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
            ItemKind::Trait(..) => Target::Trait,
            ItemKind::TraitAlias(..) => Target::TraitAlias,
            ItemKind::Impl(imp_) => Target::Impl { of_trait: imp_.of_trait.is_some() },
        }
    }

    // FIXME: For now, should only be used with def_kinds from ItemIds
    pub fn from_def_kind(def_kind: DefKind) -> Target {
        match def_kind {
            DefKind::ExternCrate => Target::ExternCrate,
            DefKind::Use => Target::Use,
            DefKind::Static { .. } => Target::Static,
            DefKind::Const => Target::Const,
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

    pub fn from_ast_item(item: &ast::Item) -> Target {
        match item.kind {
            ast::ItemKind::ExternCrate(..) => Target::ExternCrate,
            ast::ItemKind::Use(..) => Target::Use,
            ast::ItemKind::Static { .. } => Target::Static,
            ast::ItemKind::Const(..) => Target::Const,
            ast::ItemKind::Fn { .. } => Target::Fn,
            ast::ItemKind::Mod(..) => Target::Mod,
            ast::ItemKind::ForeignMod { .. } => Target::ForeignMod,
            ast::ItemKind::GlobalAsm { .. } => Target::GlobalAsm,
            ast::ItemKind::TyAlias(..) => Target::TyAlias,
            ast::ItemKind::Enum(..) => Target::Enum,
            ast::ItemKind::Struct(..) => Target::Struct,
            ast::ItemKind::Union(..) => Target::Union,
            ast::ItemKind::Trait(..) => Target::Trait,
            ast::ItemKind::TraitAlias(..) => Target::TraitAlias,
            ast::ItemKind::Impl(ref i) => Target::Impl { of_trait: i.of_trait.is_some() },
            ast::ItemKind::MacCall(..) => Target::MacroCall,
            ast::ItemKind::MacroDef(..) => Target::MacroDef,
            ast::ItemKind::Delegation(..) => Target::Delegation { mac: false },
            ast::ItemKind::DelegationMac(..) => Target::Delegation { mac: true },
        }
    }

    pub fn from_foreign_item_kind(kind: &ast::ForeignItemKind) -> Target {
        match kind {
            ForeignItemKind::Static(_) => Target::ForeignStatic,
            ForeignItemKind::Fn(_) => Target::ForeignFn,
            ForeignItemKind::TyAlias(_) => Target::ForeignTy,
            ForeignItemKind::MacCall(_) => Target::MacroCall,
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

    pub fn from_assoc_item_kind(kind: &ast::AssocItemKind, assoc_ctxt: AssocCtxt) -> Target {
        match kind {
            AssocItemKind::Const(_) => Target::AssocConst,
            AssocItemKind::Fn(f) => Target::Method(match assoc_ctxt {
                AssocCtxt::Trait => MethodKind::Trait { body: f.body.is_some() },
                AssocCtxt::Impl { of_trait } => {
                    if of_trait {
                        MethodKind::TraitImpl
                    } else {
                        MethodKind::Inherent
                    }
                }
            }),
            AssocItemKind::Type(_) => Target::AssocTy,
            AssocItemKind::Delegation(_) => Target::Delegation { mac: false },
            AssocItemKind::DelegationMac(_) => Target::Delegation { mac: true },
            AssocItemKind::MacCall(_) => Target::MacroCall,
        }
    }

    pub fn from_expr(expr: &ast::Expr) -> Self {
        match &expr.kind {
            ast::ExprKind::Closure(..) | ast::ExprKind::Gen(..) => Self::Closure,
            ast::ExprKind::Paren(e) => Self::from_expr(&e),
            _ => Self::Expression,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Target::ExternCrate => "extern crate",
            Target::Use => "use",
            Target::Static => "static",
            Target::Const => "constant",
            Target::Fn => "function",
            Target::Closure => "closure",
            Target::Mod => "module",
            Target::ForeignMod => "foreign module",
            Target::GlobalAsm => "global asm",
            Target::TyAlias => "type alias",
            Target::Enum => "enum",
            Target::Variant => "enum variant",
            Target::Struct => "struct",
            Target::Field => "struct field",
            Target::Union => "union",
            Target::Trait => "trait",
            Target::TraitAlias => "trait alias",
            Target::Impl { .. } => "implementation block",
            Target::Expression => "expression",
            Target::Statement => "statement",
            Target::Arm => "match arm",
            Target::AssocConst => "associated const",
            Target::Method(kind) => match kind {
                MethodKind::Inherent => "inherent method",
                MethodKind::Trait { body: false } => "required trait method",
                MethodKind::Trait { body: true } => "provided trait method",
                MethodKind::TraitImpl => "trait method in an impl block",
            },
            Target::AssocTy => "associated type",
            Target::ForeignFn => "foreign function",
            Target::ForeignStatic => "foreign static item",
            Target::ForeignTy => "foreign type",
            Target::GenericParam { kind, .. } => match kind {
                GenericParamKind::Type => "type parameter",
                GenericParamKind::Lifetime => "lifetime parameter",
                GenericParamKind::Const => "const parameter",
            },
            Target::MacroDef => "macro def",
            Target::Param => "function param",
            Target::PatField => "pattern field",
            Target::ExprField => "struct field",
            Target::WherePredicate => "where predicate",
            Target::MacroCall => "macro call",
            Target::Crate => "crate",
            Target::Delegation { .. } => "delegation",
        }
    }

    pub fn plural_name(self) -> &'static str {
        match self {
            Target::ExternCrate => "extern crates",
            Target::Use => "use statements",
            Target::Static => "statics",
            Target::Const => "constants",
            Target::Fn => "functions",
            Target::Closure => "closures",
            Target::Mod => "modules",
            Target::ForeignMod => "foreign modules",
            Target::GlobalAsm => "global asms",
            Target::TyAlias => "type aliases",
            Target::Enum => "enums",
            Target::Variant => "enum variants",
            Target::Struct => "structs",
            Target::Field => "struct fields",
            Target::Union => "unions",
            Target::Trait => "traits",
            Target::TraitAlias => "trait aliases",
            Target::Impl { of_trait: false } => "inherent impl blocks",
            Target::Impl { of_trait: true } => "trait impl blocks",
            Target::Expression => "expressions",
            Target::Statement => "statements",
            Target::Arm => "match arms",
            Target::AssocConst => "associated consts",
            Target::Method(kind) => match kind {
                MethodKind::Inherent => "inherent methods",
                MethodKind::Trait { body: false } => "required trait methods",
                MethodKind::Trait { body: true } => "provided trait methods",
                MethodKind::TraitImpl => "trait methods in impl blocks",
            },
            Target::AssocTy => "associated types",
            Target::ForeignFn => "foreign functions",
            Target::ForeignStatic => "foreign statics",
            Target::ForeignTy => "foreign types",
            Target::GenericParam { kind, has_default: _ } => match kind {
                GenericParamKind::Type => "type parameters",
                GenericParamKind::Lifetime => "lifetime parameters",
                GenericParamKind::Const => "const parameters",
            },
            Target::MacroDef => "macro defs",
            Target::Param => "function params",
            Target::PatField => "pattern fields",
            Target::ExprField => "struct fields",
            Target::WherePredicate => "where predicates",
            Target::MacroCall => "macro calls",
            Target::Crate => "crates",
            Target::Delegation { .. } => "delegations",
        }
    }
}
