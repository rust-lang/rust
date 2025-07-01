use rustc_ast as ast;

/// Describes what context an [`AttrTarget`] is in, like a trait implementation for example.
#[derive(Copy, Clone, Debug)]
pub enum Position {
    Free,
    Trait,
    TraitImpl,
    Impl,
    Extern,
}

/// Enumeration of things that an attribute can be on. Used during attribute parsing.
#[derive(Copy, Clone, Debug)]
pub enum AttrTarget<'ast> {
    /// ```rust,ignore (pseudo-Rust)
    /// #[attr]
    /// fn function(){}
    /// ```
    Function { pos: Position, f: &'ast ast::Fn },
    /// ```rust,ignore (pseudo-Rust)
    /// #[attr]
    /// const X: u32 = 42;
    /// ```
    Const { pos: Position },
    /// ```rust,ignore (pseudo-Rust)
    /// #[attr]
    /// type X = Y;
    /// ```
    TypeAlias { pos: Position },
    /// ```rust,ignore (pseudo-Rust)
    /// #[attr]
    /// static X: u32 = 42;
    /// ```
    Static { pos: Position },
    /// ```rust,ignore (pseudo-Rust)
    /// #[attr]
    /// reuse this::that
    /// ```
    Delegation { pos: Position },
    /// ```rust,ignore (pseudo-Rust)
    /// struct Foo {
    ///     #[attr]
    ///     field: u32,
    /// }
    /// ```
    Field,
    /// A field in a struct expression.
    /// ```rust,ignore (pseudo-Rust)
    /// Foo { #[attr] field: value }
    /// ```
    ExprField,
    /// A field in a struct pattern.
    /// ```rust,ignore (pseudo-Rust)
    /// Foo { #[attr] field }
    /// ```
    PatField,
    /// ```rust,ignore (pseudo-Rust)
    /// #[attr]
    /// enum Foo {
    ///     Variant,
    /// }
    /// ```
    Enum,
    /// ```rust,ignore (pseudo-Rust)
    /// enum Foo {
    ///     #[attr]
    ///     Variant,
    /// }
    /// ```
    EnumVariant,
    /// ```rust,ignore (pseudo-Rust)
    /// #[attr]
    /// struct Foo {
    ///     field: u32,
    /// }
    /// ```
    Struct,
    /// ```rust,ignore (pseudo-Rust)
    /// #[attr]
    /// union Foo {
    ///     field: u32,
    /// }
    /// ```
    Union,
    /// ```rust,ignore (pseudo-Rust)
    /// #[attr]
    /// extern crate foo;
    /// ```
    ExternCrate,
    /// ```rust,ignore (pseudo-Rust)
    /// #[attr]
    /// use std::vec::Vec;
    /// ```
    Use,
    /// ```rust,ignore (pseudo-Rust)
    /// #[attr]
    /// mod foo;
    /// //or
    /// #[attr]
    /// mod bar {};
    /// ```
    Module,
    /// ```rust,ignore (pseudo-Rust)
    /// #[attr]
    /// extern "C" { .. }
    /// ```
    ExternBlock,
    /// ```rust,ignore (pseudo-Rust)
    /// struct Foo<#[attr] 'a, #[attr] T, #[attr] const N: usize> {
    ///     field: &'a [T; N],
    /// }
    /// ```
    GenericParam { kind: &'ast ast::GenericParamKind },
    /// ```rust,ignore (pseudo-Rust)
    /// #![feature(where_clause_attrs)]
    ///
    /// struct A<T>
    /// where
    ///     #[attr]
    ///     T: Iterator,
    /// {
    ///     f: T,
    /// }
    /// ```
    WherePredicate,
    /// Expressions, like `#[attr] || {}`, `#[attr][a, b, c]` or `#[attr] { ... }`;
    Expression {
        /// The ast of this expression, if it exists (or not, if it is the result of a desugaring for example).
        kind: Option<&'ast ast::ExprKind>,
    },
    /// A `let` statement, like `#[attr] let x = 42;`;
    Let,
    /// ```rust,ignore (pseudo-Rust)
    /// #[attr]
    /// trait Foo {}
    /// ```
    Trait { trait_: &'ast ast::Trait },
    /// ```rust,ignore (pseudo-Rust)
    /// #[attr]
    /// macro_rules! identity {
    ///    ( $($tokens:tt)* ) => { $($tokens)* }
    /// }
    /// ```
    Macro,
    /// ```rust,ignore (pseudo-Rust)
    /// #[attr]
    /// trait Bar = Foo + Sync;
    /// ```
    TraitAlias,
    /// ```rust,ignore (pseudo-Rust)
    /// #[attr]
    /// impl Foo { .. }
    /// ```
    Impl,
    /// ```rust,ignore (pseudo-Rust)
    /// #[attr]
    /// impl Trait for Foo {  }
    /// ```
    TraitImpl,
    /// ```rust,ignore (pseudo-Rust)
    /// #[attr]
    /// core::arch::global_asm!("/* {} */", const 0);
    /// ```
    GlobalAsm,
    /// ```rust,ignore (pseudo-Rust)
    /// match 123 {
    ///    #[attr]
    ///    0..=10 => { println!("match!") },
    ///    _ => { println!("no match!") },
    /// }
    /// ```
    Arm,
    /// The crate:
    /// ```rust,ignore (pseudo-Rust)
    /// #![attr]
    /// ```
    Crate,
    /// ```rust,ignore (pseudo-Rust)
    /// fn function(#[attr] x: usize){}
    /// ```
    Param,
}

impl AttrTarget<'_> {
    pub fn from_expr(expr: &ast::Expr) -> AttrTarget<'_> {
        AttrTarget::Expression { kind: Some(&expr.kind) }
    }

    pub fn from_item(item: &ast::ItemKind) -> AttrTarget<'_> {
        use rustc_ast::ItemKind;
        let pos = Position::Free;
        match item {
            ItemKind::ExternCrate(..) => AttrTarget::ExternCrate,
            ItemKind::Use(..) => AttrTarget::Use,
            ItemKind::Static(..) => AttrTarget::Static { pos },
            ItemKind::Const(..) => AttrTarget::Const { pos },
            ItemKind::Fn(f) => AttrTarget::Function { pos, f },
            ItemKind::Mod(..) => AttrTarget::Module,
            ItemKind::ForeignMod(..) => AttrTarget::ExternBlock,
            ItemKind::GlobalAsm(..) => AttrTarget::GlobalAsm,
            ItemKind::TyAlias(..) => AttrTarget::TypeAlias { pos },
            ItemKind::Enum(..) => AttrTarget::Enum,
            ItemKind::Struct(..) => AttrTarget::Struct,
            ItemKind::Union(..) => AttrTarget::Union,
            ItemKind::Trait(trait_) => AttrTarget::Trait { trait_ },
            ItemKind::TraitAlias(..) => AttrTarget::TraitAlias,
            ItemKind::MacroDef(..) => AttrTarget::Macro,
            ItemKind::Impl { .. } => AttrTarget::Impl,
            ItemKind::Delegation(..) => AttrTarget::Delegation { pos },
            ItemKind::MacCall(..) | ItemKind::DelegationMac(..) => {
                unreachable!("macros have been expanded")
            }
        }
    }

    pub fn from_generic(generic: &ast::GenericParam) -> AttrTarget<'_> {
        AttrTarget::GenericParam { kind: &generic.kind }
    }

    pub fn from_impl_item(pos: Position, item: &ast::AssocItem) -> AttrTarget<'_> {
        use ast::AssocItemKind::*;
        match &item.kind {
            Const(_) => AttrTarget::Const { pos },
            Fn(f) => AttrTarget::Function { pos, f },
            Type(_) => AttrTarget::TypeAlias { pos },
            Delegation(_) => AttrTarget::Delegation { pos },
            MacCall(_) | DelegationMac(_) => unreachable!("macros have been expanded"),
        }
    }

    pub fn from_trait_item(item: &ast::AssocItem) -> AttrTarget<'_> {
        let pos = Position::Trait;
        use ast::AssocItemKind::*;
        match &item.kind {
            Const(_) => AttrTarget::Const { pos },
            Fn(f) => AttrTarget::Function { pos, f },
            Type(_) => AttrTarget::TypeAlias { pos },
            Delegation(_) => AttrTarget::Delegation { pos },
            MacCall(_) | DelegationMac(_) => unreachable!("macros have been expanded"),
        }
    }

    pub fn from_foreign(param: &ast::ForeignItem) -> AttrTarget<'_> {
        use ast::ForeignItemKind::*;

        let pos = Position::Extern;
        match &param.kind {
            Static(_) => AttrTarget::Const { pos },
            Fn(f) => AttrTarget::Function { pos, f },
            TyAlias(_) => AttrTarget::TypeAlias { pos },
            MacCall(_) => unreachable!("macros have been expanded"),
        }
    }
}
