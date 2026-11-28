//! This module lists attribute targets, with conversions from other types.

use std::fmt::{self, Display};

use rustc_abi::ExternAbi;
pub use rustc_ast::visit::AssocCtxt;
use rustc_ast::{
    Arm, AssocItemKind, Closure, ConstBlockItem, ConstItem, Crate, EnumDef, Expr, ExprField,
    FieldDef, Fn, ForeignItemKind, ForeignMod, GenericParam, GenericParamKind, Generics, Impl,
    InlineAsm, MacroDef, ModKind, Param, Pat, Safety, StaticItem, Stmt, Trait, TraitAlias, TyAlias,
    UseTree, Variant, VariantData, WherePredicate, ast,
};
use rustc_macros::StableHash;
use rustc_span::{Ident, Symbol};

/// This enum lists all possible types of AST items.
#[derive(Clone, Copy, Debug)]
pub enum AstTarget<'a> {
    // Target types that may correspond to different kinds of items.
    Delegation {
        mac: bool,
    },
    MacroCall,

    // Target types that correspond exclusively to `AssocItem` kind. Mapping is obtained from `Target::from_assoc_item_kind`.
    AssocConst(&'a ConstItem, AssocCtxt),
    Method(&'a Fn, MethodKind),
    AssocTy(&'a TyAlias, AssocCtxt),

    // Target types that correspond exclusively to `ForeignItem` kind. Mapping is obtained from `Target::from_foreign_item_kind`.
    ForeignStatic(&'a StaticItem),
    ForeignFn(&'a Fn),
    ForeignTy(&'a TyAlias),

    // Target types that correspond exclusively to `Item` kind. Mapping is obtained from `Target::from_ast_item`.
    ExternCrate(&'a Option<Symbol>, &'a Ident),
    Use(&'a UseTree),
    Static(&'a StaticItem),
    Const(ConstTypeAstTarget<'a>),
    Fn(&'a Fn),
    Mod(&'a Safety, &'a Ident, &'a ModKind),
    ForeignMod(&'a ForeignMod),
    GlobalAsm(&'a InlineAsm),
    TyAlias(&'a TyAlias),
    Enum(&'a Ident, &'a Generics, &'a EnumDef),
    Struct(&'a Ident, &'a Generics, &'a VariantData),
    Union(&'a Ident, &'a Generics, &'a VariantData),
    Trait(&'a Trait),
    TraitAlias(&'a TraitAlias),
    Impl(&'a Impl),
    MacroDef(&'a Ident, &'a MacroDef),

    // Target types that correspond exclusively to `Expr` kind. Mapping is obtained from `Target::from_expr`.
    Closure(&'a Closure),
    Expression(&'a Expr),

    Arm(&'a Arm),
    Crate(&'a Crate),
    ExprField(&'a ExprField),
    Field(&'a FieldDef),
    GenericParam(&'a GenericParam),
    Param(&'a Param),
    PatField(&'a Pat),
    Statement(&'a Stmt),
    Variant(&'a Variant),
    WherePredicate(&'a WherePredicate),

    /// Used when it is not possible to obtain detailed AST information.
    /// In this case, None is just a wrapper for `Target`.
    None(Target),
}

#[derive(Clone, Copy, Debug)]
pub enum ConstTypeAstTarget<'a> {
    ConstItem(&'a ConstItem),
    ConstBlockItem(&'a ConstBlockItem),
}

#[derive(Copy, Clone, Hash, PartialEq, Debug, Eq, StableHash)]
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

// FIXME(rtjkro): `AstTarget` has nearly one-to-one mapping with `Target`, barring the extra fields from `AssocConst`, `Method`, and `AssocTy`.
// In the future, remove `Target` and use `AstTarget` instead.
#[derive(Copy, Clone, Hash, PartialEq, Debug, Eq, StableHash)]
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
    AssocConst(AssocCtxt),
    Method(MethodKind),
    AssocTy(AssocCtxt),
    ForeignFn,
    ForeignStatic,
    ForeignTy,
    LifetimeParam,
    TypeParam,
    ConstParam,
    MacroDef,
    Param,
    PatField,
    ExprField,
    WherePredicate,
    MacroCall,
    Crate,
    Delegation { mac: bool },
    ForLoop,
    While,
    Loop,
    Break,
}

impl<'a> AstTarget<'a> {
    pub fn get_abi(&self) -> Option<ExternAbi> {
        let ext = match self {
            AstTarget::Method(fn_item, _) => fn_item.sig.header.ext,
            AstTarget::Fn(fn_item) => fn_item.sig.header.ext,
            AstTarget::ForeignFn(fn_item) => fn_item.sig.header.ext,
            _ => return None,
        };

        match ext {
            ast::Extern::None => Some(ExternAbi::Rust),
            ast::Extern::Implicit(_) => Some(ExternAbi::FALLBACK),
            ast::Extern::Explicit(abi, _) => Some(abi.symbol_unescaped.as_str().parse().ok()?),
        }
    }

    pub fn get_fn_sig(&self) -> Option<&rustc_ast::ast::FnSig> {
        match self {
            AstTarget::Method(fn_item, _) => Some(&fn_item.sig),
            AstTarget::Fn(fn_item) => Some(&fn_item.sig),
            AstTarget::ForeignFn(fn_item) => Some(&fn_item.sig),
            _ => None,
        }
    }

    pub fn from_foreign_item_kind(kind: &'a ast::ForeignItemKind) -> Self {
        match kind {
            ForeignItemKind::Static(static_item) => AstTarget::ForeignStatic(static_item),
            ForeignItemKind::Fn(f) => AstTarget::ForeignFn(f),
            ForeignItemKind::TyAlias(ty_alias) => AstTarget::ForeignTy(ty_alias),
            ForeignItemKind::MacCall(_) => AstTarget::MacroCall,
        }
    }

    pub fn from_assoc_item_kind(kind: &'a ast::AssocItemKind, assoc_ctxt: AssocCtxt) -> Self {
        match kind {
            AssocItemKind::Const(const_item) => AstTarget::AssocConst(const_item, assoc_ctxt),
            AssocItemKind::Fn(f) => AstTarget::Method(
                f,
                match assoc_ctxt {
                    AssocCtxt::Trait => MethodKind::Trait { body: f.body.is_some() },
                    AssocCtxt::Impl { of_trait, .. } => {
                        if of_trait {
                            MethodKind::TraitImpl
                        } else {
                            MethodKind::Inherent
                        }
                    }
                },
            ),
            AssocItemKind::Type(type_alias) => AstTarget::AssocTy(type_alias, assoc_ctxt),
            AssocItemKind::Delegation(_) => AstTarget::Delegation { mac: false },
            AssocItemKind::DelegationMac(_) => AstTarget::Delegation { mac: true },
            AssocItemKind::MacCall(_) => AstTarget::MacroCall,
        }
    }

    pub fn from_ast_item(kind: &'a ast::ItemKind) -> Self {
        match kind {
            ast::ItemKind::ExternCrate(symbol, ident) => AstTarget::ExternCrate(symbol, ident),
            ast::ItemKind::Use(use_tree) => AstTarget::Use(use_tree),
            ast::ItemKind::Static(static_item) => AstTarget::Static(static_item),
            ast::ItemKind::Const(const_item) => {
                AstTarget::Const(ConstTypeAstTarget::ConstItem(const_item))
            }
            ast::ItemKind::ConstBlock(const_block_item) => {
                AstTarget::Const(ConstTypeAstTarget::ConstBlockItem(const_block_item))
            }
            ast::ItemKind::Fn(f) => AstTarget::Fn(f),
            ast::ItemKind::Mod(s, i, mk) => AstTarget::Mod(s, i, mk),
            ast::ItemKind::ForeignMod(foreign_mod) => AstTarget::ForeignMod(foreign_mod),
            ast::ItemKind::GlobalAsm(inline_asm) => AstTarget::GlobalAsm(inline_asm),
            ast::ItemKind::TyAlias(ty_alias) => AstTarget::TyAlias(ty_alias),
            ast::ItemKind::Enum(i, g, ed) => AstTarget::Enum(i, g, ed),
            ast::ItemKind::Struct(i, g, vd) => AstTarget::Struct(i, g, vd),
            ast::ItemKind::Union(i, g, vd) => AstTarget::Union(i, g, vd),
            ast::ItemKind::Trait(trait_kind) => AstTarget::Trait(trait_kind),
            ast::ItemKind::TraitAlias(trait_alias) => AstTarget::TraitAlias(trait_alias),
            ast::ItemKind::Impl(i) => AstTarget::Impl(i),
            ast::ItemKind::MacCall(..) => AstTarget::MacroCall,
            ast::ItemKind::MacroDef(ident, macro_def) => AstTarget::MacroDef(ident, macro_def),
            ast::ItemKind::Delegation(..) => AstTarget::Delegation { mac: false },
            ast::ItemKind::DelegationMac(..) => AstTarget::Delegation { mac: true },
            ast::ItemKind::TestBinderConstraints(..) => AstTarget::MacroCall,
        }
    }
}

impl Display for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", Self::name(*self))
    }
}

rustc_error_messages::into_diag_arg_using_display!(Target);

impl Target {
    pub fn from_ast_target(ast_item: &AstTarget<'_>) -> Self {
        match ast_item {
            AstTarget::Delegation { mac } => Target::Delegation { mac: *mac },
            AstTarget::MacroCall => Target::MacroCall,

            AstTarget::AssocConst(_, assoc_ctxt) => Target::AssocConst(*assoc_ctxt),
            AstTarget::Method(_, method_kind) => Target::Method(*method_kind),
            AstTarget::AssocTy(_, assoc_ctxt) => Target::AssocTy(*assoc_ctxt),

            AstTarget::ForeignStatic(_) => Target::ForeignStatic,
            AstTarget::ForeignFn(_) => Target::ForeignFn,
            AstTarget::ForeignTy(_) => Target::ForeignTy,

            AstTarget::ExternCrate(..) => Target::ExternCrate,
            AstTarget::Use(_) => Target::Use,
            AstTarget::Static(_) => Target::Static,
            AstTarget::Const(_) => Target::Const,
            AstTarget::Fn(_) => Target::Fn,
            AstTarget::Mod(..) => Target::Mod,
            AstTarget::ForeignMod(_) => Target::ForeignMod,
            AstTarget::GlobalAsm(_) => Target::GlobalAsm,
            AstTarget::TyAlias(_) => Target::TyAlias,
            AstTarget::Enum(..) => Target::Enum,
            AstTarget::Struct(..) => Target::Struct,
            AstTarget::Union(..) => Target::Union,
            AstTarget::Trait(_) => Target::Trait,
            AstTarget::TraitAlias(_) => Target::TraitAlias,
            AstTarget::Impl(i) => Target::Impl { of_trait: i.of_trait.is_some() },
            AstTarget::MacroDef(..) => Target::MacroDef,

            AstTarget::Closure(_) => Target::Closure,
            AstTarget::Expression(e) => Target::from_expr(e),

            AstTarget::Arm(_) => Target::Arm,
            AstTarget::Crate(_) => Target::Crate,
            AstTarget::ExprField(_) => Target::ExprField,
            AstTarget::Field(_) => Target::Field,
            AstTarget::GenericParam(g) => match g.kind {
                GenericParamKind::Type { .. } => Target::TypeParam,
                GenericParamKind::Lifetime => Target::LifetimeParam,
                GenericParamKind::Const { .. } => Target::ConstParam,
            },
            AstTarget::Param(_) => Target::Param,
            AstTarget::PatField(_) => Target::PatField,
            AstTarget::Statement(_) => Target::Statement,
            AstTarget::Variant(_) => Target::Variant,
            AstTarget::WherePredicate(_) => Target::WherePredicate,

            AstTarget::None(target) => *target,
        }
    }

    pub fn is_associated_item(self) -> bool {
        match self {
            Target::AssocConst(_) | Target::AssocTy(_) | Target::Method(_) => true,
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
            | Target::TypeParam
            | Target::LifetimeParam
            | Target::ConstParam
            | Target::MacroDef
            | Target::Param
            | Target::PatField
            | Target::ExprField
            | Target::MacroCall
            | Target::Crate
            | Target::WherePredicate
            | Target::Delegation { .. }
            | Target::Loop
            | Target::While
            | Target::ForLoop
            | Target::Break => false,
        }
    }
    pub fn from_ast_item(item: &ast::Item) -> Target {
        match item.kind {
            ast::ItemKind::ExternCrate(..) => Target::ExternCrate,
            ast::ItemKind::Use(..) => Target::Use,
            ast::ItemKind::Static { .. } => Target::Static,
            ast::ItemKind::Const(..) => Target::Const,
            ast::ItemKind::ConstBlock(..) => Target::Const,
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
            ast::ItemKind::TestBinderConstraints(..) => Target::MacroCall,
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

    pub fn from_assoc_item_kind(kind: &ast::AssocItemKind, assoc_ctxt: AssocCtxt) -> Target {
        match kind {
            AssocItemKind::Const(_) => Target::AssocConst(assoc_ctxt),
            AssocItemKind::Fn(f) => Target::Method(match assoc_ctxt {
                AssocCtxt::Trait => MethodKind::Trait { body: f.body.is_some() },
                AssocCtxt::Impl { of_trait, .. } => {
                    if of_trait {
                        MethodKind::TraitImpl
                    } else {
                        MethodKind::Inherent
                    }
                }
            }),
            AssocItemKind::Type(_) => Target::AssocTy(assoc_ctxt),
            AssocItemKind::Delegation(_) => Target::Delegation { mac: false },
            AssocItemKind::DelegationMac(_) => Target::Delegation { mac: true },
            AssocItemKind::MacCall(_) => Target::MacroCall,
        }
    }

    pub fn from_expr(expr: &ast::Expr) -> Self {
        match &expr.kind {
            ast::ExprKind::Closure(..) | ast::ExprKind::Gen(..) => Self::Closure,
            ast::ExprKind::Paren(e) => Self::from_expr(&e),
            ast::ExprKind::ForLoop { .. } => Self::ForLoop,
            ast::ExprKind::Loop(..) => Self::Loop,
            ast::ExprKind::While(..) => Self::While,
            ast::ExprKind::Break(..) => Self::Break,
            _ => Self::Expression,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            // tidy-alphabetical-start
            Target::Arm => "match arm",
            Target::AssocConst(_) => "associated const",
            Target::AssocTy(_) => "associated type",
            Target::Break => "break expression",
            Target::Closure => "closure",
            Target::Const => "constant",
            Target::ConstParam => "const parameter",
            Target::Crate => "crate",
            Target::Delegation { .. } => "delegation",
            Target::Enum => "enum",
            Target::ExprField => "struct field",
            Target::Expression => "expression",
            Target::ExternCrate => "extern crate",
            Target::Field => "struct field",
            Target::Fn => "function",
            Target::ForLoop => "for loop",
            Target::ForeignFn => "foreign function",
            Target::ForeignMod => "foreign module",
            Target::ForeignStatic => "foreign static item",
            Target::ForeignTy => "foreign type",
            Target::GlobalAsm => "global asm",
            Target::Impl { .. } => "implementation block",
            Target::LifetimeParam => "lifetime parameter",
            Target::Loop => "loop",
            Target::MacroCall => "macro call",
            Target::MacroDef => "macro def",
            Target::Method(MethodKind::Inherent) => "inherent method",
            Target::Method(MethodKind::Trait { body: false }) => "required trait method",
            Target::Method(MethodKind::Trait { body: true }) => "provided trait method",
            Target::Method(MethodKind::TraitImpl) => "trait method in an impl block",
            Target::Mod => "module",
            Target::Param => "function param",
            Target::PatField => "pattern field",
            Target::Statement => "statement",
            Target::Static => "static",
            Target::Struct => "struct",
            Target::Trait => "trait",
            Target::TraitAlias => "trait alias",
            Target::TyAlias => "type alias",
            Target::TypeParam => "type parameter",
            Target::Union => "union",
            Target::Use => "use",
            Target::Variant => "enum variant",
            Target::WherePredicate => "where predicate",
            Target::While => "while loop",
            // tidy-alphabetical-end
        }
    }

    pub fn plural_name(self) -> &'static str {
        match self {
            // tidy-alphabetical-start
            Target::Arm => "match arms",
            Target::AssocConst(_) => "associated consts",
            Target::AssocTy(_) => "associated types",
            Target::Break => "break expressions",
            Target::Closure => "closures",
            Target::Const => "constants",
            Target::ConstParam => "const parameters",
            Target::Crate => "crates",
            Target::Delegation { .. } => "delegations",
            Target::Enum => "enums",
            Target::ExprField => "struct fields",
            Target::Expression => "expressions",
            Target::ExternCrate => "extern crates",
            Target::Field => "struct fields",
            Target::Fn => "functions",
            Target::ForLoop => "for loops",
            Target::ForeignFn => "foreign functions",
            Target::ForeignMod => "foreign modules",
            Target::ForeignStatic => "foreign statics",
            Target::ForeignTy => "foreign types",
            Target::GlobalAsm => "global asms",
            Target::Impl { of_trait: false } => "inherent impl blocks",
            Target::Impl { of_trait: true } => "trait impl blocks",
            Target::LifetimeParam => "lifetime parameters",
            Target::Loop => "loops",
            Target::MacroCall => "macro calls",
            Target::MacroDef => "macro defs",
            Target::Method(MethodKind::Inherent) => "inherent methods",
            Target::Method(MethodKind::Trait { body: false }) => "required trait methods",
            Target::Method(MethodKind::Trait { body: true }) => "provided trait methods",
            Target::Method(MethodKind::TraitImpl) => "trait methods in impl blocks",
            Target::Mod => "modules",
            Target::Param => "function params",
            Target::PatField => "pattern fields",
            Target::Statement => "statements",
            Target::Static => "statics",
            Target::Struct => "structs",
            Target::Trait => "traits",
            Target::TraitAlias => "trait aliases",
            Target::TyAlias => "type aliases",
            Target::TypeParam => "type parameters",
            Target::Union => "unions",
            Target::Use => "use statements",
            Target::Variant => "enum variants",
            Target::WherePredicate => "where predicates",
            Target::While => "while loops",
            // tidy-alphabetical-end
        }
    }
}
