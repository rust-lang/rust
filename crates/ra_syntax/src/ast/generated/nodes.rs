//! Generated file, do not edit by hand, see `xtask/src/codegen`

use crate::{
    ast::{self, support, AstChildren, AstNode},
    SyntaxKind::{self, *},
    SyntaxNode, SyntaxToken, T,
};
/// The entire Rust source file. Includes all top-level inner attributes and module items.
///
/// [Reference](https://doc.rust-lang.org/reference/crates-and-source-files.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SourceFile {
    pub(crate) syntax: SyntaxNode,
}
impl ast::ModuleItemOwner for SourceFile {}
impl ast::AttrsOwner for SourceFile {}
impl ast::DocCommentsOwner for SourceFile {}
impl SourceFile {
    pub fn modules(&self) -> AstChildren<Module> { support::children(&self.syntax) }
}
/// Function definition either with body or not.
/// Includes all of its attributes and doc comments.
///
/// ```
/// ❰
///     /// Docs
///     #[attr]
///     pub extern "C" fn foo<T>(#[attr] Patern {p}: Pattern) -> u32
///     where
///         T: Debug
///     {
///         42
///     }
/// ❱
///
/// extern "C" {
///     ❰ fn fn_decl(also_variadic_ffi: u32, ...) -> u32; ❱
/// }
/// ```
///
/// - [Reference](https://doc.rust-lang.org/reference/items/functions.html)
/// - [Nomicon](https://doc.rust-lang.org/nomicon/ffi.html#variadic-functions)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FnDef {
    pub(crate) syntax: SyntaxNode,
}
impl ast::VisibilityOwner for FnDef {}
impl ast::NameOwner for FnDef {}
impl ast::TypeParamsOwner for FnDef {}
impl ast::DocCommentsOwner for FnDef {}
impl ast::AttrsOwner for FnDef {}
impl FnDef {
    pub fn abi(&self) -> Option<Abi> { support::child(&self.syntax) }
    pub fn const_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![const]) }
    pub fn default_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![default]) }
    pub fn async_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![async]) }
    pub fn unsafe_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![unsafe]) }
    pub fn fn_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![fn]) }
    pub fn param_list(&self) -> Option<ParamList> { support::child(&self.syntax) }
    pub fn ret_type(&self) -> Option<RetType> { support::child(&self.syntax) }
    pub fn body(&self) -> Option<BlockExpr> { support::child(&self.syntax) }
    pub fn semicolon_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![;]) }
}
/// Return type annotation.
///
/// ```
/// fn foo(a: u32) ❰ -> Option<u32> ❱ { Some(a) }
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/items/functions.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RetType {
    pub(crate) syntax: SyntaxNode,
}
impl RetType {
    pub fn thin_arrow_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![->]) }
    pub fn type_ref(&self) -> Option<TypeRef> { support::child(&self.syntax) }
}
/// Struct definition.
/// Includes all of its attributes and doc comments.
///
/// ```
/// ❰
///     /// Docs
///     #[attr]
///     struct Foo<T> where T: Debug {
///         /// Docs
///         #[attr]
///         pub a: u32,
///         b: T,
///     }
/// ❱
///
/// ❰ struct Foo; ❱
/// ❰ struct Foo<T>(#[attr] T) where T: Debug; ❱
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/items/structs.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StructDef {
    pub(crate) syntax: SyntaxNode,
}
impl ast::VisibilityOwner for StructDef {}
impl ast::NameOwner for StructDef {}
impl ast::TypeParamsOwner for StructDef {}
impl ast::AttrsOwner for StructDef {}
impl ast::DocCommentsOwner for StructDef {}
impl StructDef {
    pub fn struct_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![struct]) }
    pub fn field_def_list(&self) -> Option<FieldDefList> { support::child(&self.syntax) }
    pub fn semicolon_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![;]) }
}
/// Union definition.
/// Includes all of its attributes and doc comments.
///
/// ```
/// ❰
///     /// Docs
///     #[attr]
///     pub union Foo<T> where T: Debug {
///         /// Docs
///         #[attr]
///         a: T,
///         b: u32,
///     }
/// ❱
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/items/unions.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UnionDef {
    pub(crate) syntax: SyntaxNode,
}
impl ast::VisibilityOwner for UnionDef {}
impl ast::NameOwner for UnionDef {}
impl ast::TypeParamsOwner for UnionDef {}
impl ast::AttrsOwner for UnionDef {}
impl ast::DocCommentsOwner for UnionDef {}
impl UnionDef {
    pub fn union_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![union]) }
    pub fn record_field_def_list(&self) -> Option<RecordFieldDefList> {
        support::child(&self.syntax)
    }
}
/// Record field definition list including enclosing curly braces.
///
/// ```
/// struct Foo // same for union
/// ❰
///     {
///         a: u32,
///         b: bool,
///     }
/// ❱
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/items/structs.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RecordFieldDefList {
    pub(crate) syntax: SyntaxNode,
}
impl RecordFieldDefList {
    pub fn l_curly_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T!['{']) }
    pub fn fields(&self) -> AstChildren<RecordFieldDef> { support::children(&self.syntax) }
    pub fn r_curly_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T!['}']) }
}
/// Record field definition including its attributes and doc comments.
///
/// ` ``
/// same for union
/// struct Foo {
///      ❰
///          /// Docs
///          #[attr]
///          pub a: u32
///      ❱
///
///      ❰ b: bool ❱
/// }
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/items/structs.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RecordFieldDef {
    pub(crate) syntax: SyntaxNode,
}
impl ast::VisibilityOwner for RecordFieldDef {}
impl ast::NameOwner for RecordFieldDef {}
impl ast::AttrsOwner for RecordFieldDef {}
impl ast::DocCommentsOwner for RecordFieldDef {}
impl ast::TypeAscriptionOwner for RecordFieldDef {}
impl RecordFieldDef {}
/// Tuple field definition list including enclosing parens.
///
/// ```
/// struct Foo ❰ (u32, String, Vec<u32>) ❱;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/items/structs.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TupleFieldDefList {
    pub(crate) syntax: SyntaxNode,
}
impl TupleFieldDefList {
    pub fn l_paren_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T!['(']) }
    pub fn fields(&self) -> AstChildren<TupleFieldDef> { support::children(&self.syntax) }
    pub fn r_paren_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![')']) }
}
/// Tuple field definition including its attributes.
///
/// ```
/// struct Foo(❰ #[attr] u32 ❱);
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/items/structs.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TupleFieldDef {
    pub(crate) syntax: SyntaxNode,
}
impl ast::VisibilityOwner for TupleFieldDef {}
impl ast::AttrsOwner for TupleFieldDef {}
impl TupleFieldDef {
    pub fn type_ref(&self) -> Option<TypeRef> { support::child(&self.syntax) }
}
/// Enum definition.
/// Includes all of its attributes and doc comments.
///
/// ```
/// ❰
///     /// Docs
///     #[attr]
///     pub enum Foo<T> where T: Debug {
///         /// Docs
///         #[attr]
///         Bar,
///         Baz(#[attr] u32),
///         Bruh {
///             a: u32,
///             /// Docs
///             #[attr]
///             b: T,
///         }
///     }
/// ❱
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/items/enumerations.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EnumDef {
    pub(crate) syntax: SyntaxNode,
}
impl ast::VisibilityOwner for EnumDef {}
impl ast::NameOwner for EnumDef {}
impl ast::TypeParamsOwner for EnumDef {}
impl ast::AttrsOwner for EnumDef {}
impl ast::DocCommentsOwner for EnumDef {}
impl EnumDef {
    pub fn enum_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![enum]) }
    pub fn variant_list(&self) -> Option<EnumVariantList> { support::child(&self.syntax) }
}
/// Enum variant definition list including enclosing curly braces.
///
/// ```
/// enum Foo
/// ❰
///     {
///         Bar,
///         Baz(u32),
///         Bruh {
///             a: u32
///         }
///     }
/// ❱
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/items/enumerations.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EnumVariantList {
    pub(crate) syntax: SyntaxNode,
}
impl EnumVariantList {
    pub fn l_curly_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T!['{']) }
    pub fn variants(&self) -> AstChildren<EnumVariant> { support::children(&self.syntax) }
    pub fn r_curly_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T!['}']) }
}
/// Enum variant definition including its attributes and discriminant value definition.
///
/// ```
/// enum Foo {
///     ❰
///         /// Docs
///         #[attr]
///         Bar
///     ❱
///
///     // same for tuple and record variants
/// }
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/items/enumerations.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EnumVariant {
    pub(crate) syntax: SyntaxNode,
}
impl ast::VisibilityOwner for EnumVariant {}
impl ast::NameOwner for EnumVariant {}
impl ast::DocCommentsOwner for EnumVariant {}
impl ast::AttrsOwner for EnumVariant {}
impl EnumVariant {
    pub fn field_def_list(&self) -> Option<FieldDefList> { support::child(&self.syntax) }
    pub fn eq_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![=]) }
    pub fn expr(&self) -> Option<Expr> { support::child(&self.syntax) }
}
/// Trait definition.
/// Includes all of its attributes and doc comments.
///
/// ```
/// ❰
///     /// Docs
///     #[attr]
///     pub unsafe trait Foo<T>: Debug where T: Debug {
///         // ...
///     }
/// ❱
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/items/traits.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TraitDef {
    pub(crate) syntax: SyntaxNode,
}
impl ast::VisibilityOwner for TraitDef {}
impl ast::NameOwner for TraitDef {}
impl ast::AttrsOwner for TraitDef {}
impl ast::DocCommentsOwner for TraitDef {}
impl ast::TypeParamsOwner for TraitDef {}
impl ast::TypeBoundsOwner for TraitDef {}
impl TraitDef {
    pub fn unsafe_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![unsafe]) }
    pub fn auto_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![auto]) }
    pub fn trait_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![trait]) }
    pub fn item_list(&self) -> Option<ItemList> { support::child(&self.syntax) }
}
/// Module definition either with body or not.
/// Includes all of its inner and outer attributes, module items, doc comments.
///
/// ```
/// ❰
///     /// Docs
///     #[attr]
///     pub mod foo;
/// ❱
///
/// ❰
///     /// Docs
///     #[attr]
///     pub mod bar {
///        //! Inner docs
///        #![inner_attr]
///     }
/// ❱
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/items/modules.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Module {
    pub(crate) syntax: SyntaxNode,
}
impl ast::VisibilityOwner for Module {}
impl ast::NameOwner for Module {}
impl ast::AttrsOwner for Module {}
impl ast::DocCommentsOwner for Module {}
impl Module {
    pub fn mod_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![mod]) }
    pub fn item_list(&self) -> Option<ItemList> { support::child(&self.syntax) }
    pub fn semicolon_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![;]) }
}
/// Item defintion list.
/// This is used for both top-level items and impl block items.
///
/// ```
/// ❰
///     fn foo {}
///     struct Bar;
///     enum Baz;
///     trait Bruh;
///     const BRUUH: u32 = 42;
/// ❱
///
/// impl Foo
/// ❰
///     {
///         fn bar() {}
///         const BAZ: u32 = 42;
///     }
/// ❱
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/items.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ItemList {
    pub(crate) syntax: SyntaxNode,
}
impl ast::ModuleItemOwner for ItemList {}
impl ItemList {
    pub fn l_curly_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T!['{']) }
    pub fn assoc_items(&self) -> AstChildren<AssocItem> { support::children(&self.syntax) }
    pub fn r_curly_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T!['}']) }
}
/// Constant variable definition.
/// Includes all of its attributes and doc comments.
///
/// ```
/// ❰
///     /// Docs
///     #[attr]
///     pub const FOO: u32 = 42;
/// ❱
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/items/constant-items.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConstDef {
    pub(crate) syntax: SyntaxNode,
}
impl ast::VisibilityOwner for ConstDef {}
impl ast::NameOwner for ConstDef {}
impl ast::TypeParamsOwner for ConstDef {}
impl ast::AttrsOwner for ConstDef {}
impl ast::DocCommentsOwner for ConstDef {}
impl ast::TypeAscriptionOwner for ConstDef {}
impl ConstDef {
    pub fn default_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![default]) }
    pub fn const_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![const]) }
    pub fn eq_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![=]) }
    pub fn body(&self) -> Option<Expr> { support::child(&self.syntax) }
    pub fn semicolon_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![;]) }
}
/// Static variable definition.
/// Includes all of its attributes and doc comments.
///
/// ```
/// ❰
///     /// Docs
///     #[attr]
///     pub static mut FOO: u32 = 42;
/// ❱
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/items/static-items.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StaticDef {
    pub(crate) syntax: SyntaxNode,
}
impl ast::VisibilityOwner for StaticDef {}
impl ast::NameOwner for StaticDef {}
impl ast::TypeParamsOwner for StaticDef {}
impl ast::AttrsOwner for StaticDef {}
impl ast::DocCommentsOwner for StaticDef {}
impl ast::TypeAscriptionOwner for StaticDef {}
impl StaticDef {
    pub fn static_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![static]) }
    pub fn mut_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![mut]) }
    pub fn eq_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![=]) }
    pub fn body(&self) -> Option<Expr> { support::child(&self.syntax) }
    pub fn semicolon_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![;]) }
}
/// Type alias definition.
/// Includes associated type clauses with type bounds.
///
/// ```
/// ❰
///     /// Docs
///     #[attr]
///     pub type Foo<T> where T: Debug = T;
/// ❱
///
/// trait Bar {
///     ❰ type Baz: Debug; ❱
///     ❰ type Bruh = String; ❱
///     ❰ type Bruuh: Debug = u32; ❱
/// }
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/items/type-aliases.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypeAliasDef {
    pub(crate) syntax: SyntaxNode,
}
impl ast::VisibilityOwner for TypeAliasDef {}
impl ast::NameOwner for TypeAliasDef {}
impl ast::TypeParamsOwner for TypeAliasDef {}
impl ast::AttrsOwner for TypeAliasDef {}
impl ast::DocCommentsOwner for TypeAliasDef {}
impl ast::TypeBoundsOwner for TypeAliasDef {}
impl TypeAliasDef {
    pub fn default_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![default]) }
    pub fn type_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![type]) }
    pub fn eq_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![=]) }
    pub fn type_ref(&self) -> Option<TypeRef> { support::child(&self.syntax) }
    pub fn semicolon_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![;]) }
}
/// Inherent and trait impl definition.
/// Includes all of its inner and outer attributes.
///
/// ```
/// ❰
///     #[attr]
///     unsafe impl<T> const !Foo for Bar where T: Debug {
///         #![inner_attr]
///         // ...
///     }
/// ❱
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/items/implementations.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ImplDef {
    pub(crate) syntax: SyntaxNode,
}
impl ast::TypeParamsOwner for ImplDef {}
impl ast::AttrsOwner for ImplDef {}
impl ast::DocCommentsOwner for ImplDef {}
impl ImplDef {
    pub fn default_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![default]) }
    pub fn const_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![const]) }
    pub fn unsafe_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![unsafe]) }
    pub fn impl_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![impl]) }
    pub fn excl_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![!]) }
    pub fn for_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![for]) }
    pub fn item_list(&self) -> Option<ItemList> { support::child(&self.syntax) }
}
/// Parenthesized type reference.
/// Note: parens are only used for grouping, this is not a tuple type.
///
/// ```
/// // This is effectively just `u32`.
/// // Single-item tuple must be defined with a trailing comma: `(u32,)`
/// type Foo = ❰ (u32) ❱;
///
/// let bar: &'static ❰ (dyn Debug) ❱ = "bruh";
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParenType {
    pub(crate) syntax: SyntaxNode,
}
impl ParenType {
    pub fn l_paren_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T!['(']) }
    pub fn type_ref(&self) -> Option<TypeRef> { support::child(&self.syntax) }
    pub fn r_paren_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![')']) }
}
/// Unnamed tuple type.
///
/// ```
/// let foo: ❰ (u32, bool) ❱ = (42, true);
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/types/tuple.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TupleType {
    pub(crate) syntax: SyntaxNode,
}
impl TupleType {
    pub fn l_paren_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T!['(']) }
    pub fn fields(&self) -> AstChildren<TypeRef> { support::children(&self.syntax) }
    pub fn r_paren_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![')']) }
}
/// The never type (i.e. the exclamation point).
///
/// ```
/// type T = ❰ ! ❱;
///
/// fn no_return() -> ❰ ! ❱ {
///     loop {}
/// }
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/types/never.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NeverType {
    pub(crate) syntax: SyntaxNode,
}
impl NeverType {
    pub fn excl_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![!]) }
}
/// Path to a type.
/// Includes single identifier type names and elaborate paths with
/// generic parameters.
///
/// ```
/// type Foo = ❰ String ❱;
/// type Bar = ❰ std::vec::Vec<T> ❱;
/// type Baz = ❰ ::bruh::<Bruuh as Iterator>::Item ❱;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/paths.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PathType {
    pub(crate) syntax: SyntaxNode,
}
impl PathType {
    pub fn path(&self) -> Option<Path> { support::child(&self.syntax) }
}
/// Raw pointer type.
///
/// ```
/// type Foo = ❰ *const u32 ❱;
/// type Bar = ❰ *mut u32 ❱;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/types/pointer.html#raw-pointers-const-and-mut)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PointerType {
    pub(crate) syntax: SyntaxNode,
}
impl PointerType {
    pub fn star_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![*]) }
    pub fn const_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![const]) }
    pub fn mut_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![mut]) }
    pub fn type_ref(&self) -> Option<TypeRef> { support::child(&self.syntax) }
}
/// Array type.
///
/// ```
/// type Foo = ❰ [u32; 24 - 3] ❱;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/types/array.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ArrayType {
    pub(crate) syntax: SyntaxNode,
}
impl ArrayType {
    pub fn l_brack_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T!['[']) }
    pub fn type_ref(&self) -> Option<TypeRef> { support::child(&self.syntax) }
    pub fn semicolon_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![;]) }
    pub fn expr(&self) -> Option<Expr> { support::child(&self.syntax) }
    pub fn r_brack_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![']']) }
}
/// Slice type.
///
/// ```
/// type Foo = ❰ [u8] ❱;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/types/slice.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SliceType {
    pub(crate) syntax: SyntaxNode,
}
impl SliceType {
    pub fn l_brack_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T!['[']) }
    pub fn type_ref(&self) -> Option<TypeRef> { support::child(&self.syntax) }
    pub fn r_brack_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![']']) }
}
/// Reference type.
///
/// ```
/// type Foo = ❰ &'static str ❱;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/types/pointer.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ReferenceType {
    pub(crate) syntax: SyntaxNode,
}
impl ReferenceType {
    pub fn amp_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![&]) }
    pub fn lifetime_token(&self) -> Option<SyntaxToken> {
        support::token(&self.syntax, T![lifetime])
    }
    pub fn mut_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![mut]) }
    pub fn type_ref(&self) -> Option<TypeRef> { support::child(&self.syntax) }
}
/// Placeholder type (i.e. the underscore).
///
/// ```
/// let foo: ❰ _ ❱ = 42_u32;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/types/inferred.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PlaceholderType {
    pub(crate) syntax: SyntaxNode,
}
impl PlaceholderType {
    pub fn underscore_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![_]) }
}
/// Function pointer type (not to be confused with `Fn*` family of traits).
///
/// ```
/// type Foo = ❰ async fn(#[attr] u32, named: bool) -> u32 ❱;
///
/// type Bar = ❰ extern "C" fn(variadic: u32, #[attr] ...) ❱;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/types/function-pointer.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FnPointerType {
    pub(crate) syntax: SyntaxNode,
}
impl FnPointerType {
    pub fn abi(&self) -> Option<Abi> { support::child(&self.syntax) }
    pub fn unsafe_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![unsafe]) }
    pub fn fn_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![fn]) }
    pub fn param_list(&self) -> Option<ParamList> { support::child(&self.syntax) }
    pub fn ret_type(&self) -> Option<RetType> { support::child(&self.syntax) }
}
/// Higher order type.
///
/// ```
/// type Foo = ❰ for<'a> fn(&'a str) ❱;
/// ```
///
/// [Reference](https://doc.rust-lang.org/nomicon/hrtb.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ForType {
    pub(crate) syntax: SyntaxNode,
}
impl ForType {
    pub fn for_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![for]) }
    pub fn type_param_list(&self) -> Option<TypeParamList> { support::child(&self.syntax) }
    pub fn type_ref(&self) -> Option<TypeRef> { support::child(&self.syntax) }
}
/// Opaque `impl Trait` type.
///
/// ```
/// fn foo(bar: ❰ impl Debug + Eq ❱) {}
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/types/impl-trait.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ImplTraitType {
    pub(crate) syntax: SyntaxNode,
}
impl ast::TypeBoundsOwner for ImplTraitType {}
impl ImplTraitType {
    pub fn impl_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![impl]) }
}
/// Trait object type.
///
/// ```
/// type Foo = ❰ dyn Debug ❱;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/types/trait-object.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DynTraitType {
    pub(crate) syntax: SyntaxNode,
}
impl ast::TypeBoundsOwner for DynTraitType {}
impl DynTraitType {
    pub fn dyn_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![dyn]) }
}
/// Tuple literal.
///
/// ```
/// ❰ (42, true) ❱;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/expressions/tuple-expr.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TupleExpr {
    pub(crate) syntax: SyntaxNode,
}
impl ast::AttrsOwner for TupleExpr {}
impl TupleExpr {
    pub fn l_paren_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T!['(']) }
    pub fn exprs(&self) -> AstChildren<Expr> { support::children(&self.syntax) }
    pub fn r_paren_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![')']) }
}
/// Array literal.
///
/// ```
/// ❰ [#![inner_attr] true, false, true] ❱;
///
/// ❰ ["baz"; 24] ❱;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/expressions/array-expr.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ArrayExpr {
    pub(crate) syntax: SyntaxNode,
}
impl ast::AttrsOwner for ArrayExpr {}
impl ArrayExpr {
    pub fn l_brack_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T!['[']) }
    pub fn exprs(&self) -> AstChildren<Expr> { support::children(&self.syntax) }
    pub fn semicolon_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![;]) }
    pub fn r_brack_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![']']) }
}
/// Parenthesized expression.
/// Note: parens are only used for grouping, this is not a tuple literal.
///
/// ```
/// ❰ (#![inner_attr] 2 + 2) ❱ * 2;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/expressions/grouped-expr.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParenExpr {
    pub(crate) syntax: SyntaxNode,
}
impl ast::AttrsOwner for ParenExpr {}
impl ParenExpr {
    pub fn l_paren_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T!['(']) }
    pub fn expr(&self) -> Option<Expr> { support::child(&self.syntax) }
    pub fn r_paren_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![')']) }
}
/// Path to a symbol in expression context.
/// Includes single identifier variable names and elaborate paths with
/// generic parameters.
///
/// ```
/// ❰ Some::<i32> ❱;
/// ❰ foo ❱ + 42;
/// ❰ Vec::<i32>::push ❱;
/// ❰ <[i32]>::reverse ❱;
/// ❰ <String as std::borrow::Borrow<str>>::borrow ❱;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/expressions/path-expr.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PathExpr {
    pub(crate) syntax: SyntaxNode,
}
impl PathExpr {
    pub fn path(&self) -> Option<Path> { support::child(&self.syntax) }
}
/// Anonymous callable object literal a.k.a. closure, lambda or functor.
///
/// ```
/// ❰ || 42 ❱;
/// ❰ |a: u32| val + 1 ❱;
/// ❰ async |#[attr] Pattern(_): Pattern| { bar } ❱;
/// ❰ move || baz ❱;
/// ❰ || -> u32 { closure_with_ret_type_annotation_requires_block_expr } ❱
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/expressions/closure-expr.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LambdaExpr {
    pub(crate) syntax: SyntaxNode,
}
impl ast::AttrsOwner for LambdaExpr {}
impl LambdaExpr {
    pub fn static_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![static]) }
    pub fn async_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![async]) }
    pub fn move_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![move]) }
    pub fn param_list(&self) -> Option<ParamList> { support::child(&self.syntax) }
    pub fn ret_type(&self) -> Option<RetType> { support::child(&self.syntax) }
    pub fn body(&self) -> Option<Expr> { support::child(&self.syntax) }
}
/// If expression. Includes both regular `if` and `if let` forms.
/// Beware that `else if` is a special case syntax sugar, because in general
/// there has to be block expression after `else`.
///
/// ```
/// ❰ if bool_cond { 42 } ❱
/// ❰ if bool_cond { 42 } else { 24 } ❱
/// ❰ if bool_cond { 42 } else if bool_cond2 { 42 } ❱
///
/// ❰
///     if let Pattern(foo) = bar {
///         foo
///     } else {
///         panic!();
///     }
/// ❱
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/expressions/if-expr.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IfExpr {
    pub(crate) syntax: SyntaxNode,
}
impl ast::AttrsOwner for IfExpr {}
impl IfExpr {
    pub fn if_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![if]) }
    pub fn condition(&self) -> Option<Condition> { support::child(&self.syntax) }
}
/// Unconditional loop expression.
///
/// ```
/// ❰
///     loop {
///         // yeah, it's that simple...
///     }
/// ❱
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/expressions/loop-expr.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LoopExpr {
    pub(crate) syntax: SyntaxNode,
}
impl ast::AttrsOwner for LoopExpr {}
impl ast::LoopBodyOwner for LoopExpr {}
impl LoopExpr {
    pub fn loop_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![loop]) }
}
/// Block expression with an optional prefix (label, try ketword,
/// unsafe keyword, async keyword...).
///
/// ```
/// ❰
///     'label: try {
///         None?
///     }
/// ❱
/// ```
///
/// - [try block](https://doc.rust-lang.org/unstable-book/language-features/try-blocks.html)
/// - [unsafe block](https://doc.rust-lang.org/reference/expressions/block-expr.html#unsafe-blocks)
/// - [async block](https://doc.rust-lang.org/reference/expressions/block-expr.html#async-blocks)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EffectExpr {
    pub(crate) syntax: SyntaxNode,
}
impl ast::AttrsOwner for EffectExpr {}
impl EffectExpr {
    pub fn label(&self) -> Option<Label> { support::child(&self.syntax) }
    pub fn try_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![try]) }
    pub fn unsafe_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![unsafe]) }
    pub fn async_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![async]) }
    pub fn block_expr(&self) -> Option<BlockExpr> { support::child(&self.syntax) }
}
/// For loop expression.
/// Note: record struct literals are not valid as iterable expression
/// due to ambiguity.
///
/// ```
/// ❰
/// for i in (0..4) {
///     dbg!(i);
/// }
/// ❱
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/expressions/loop-expr.html#iterator-loops)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ForExpr {
    pub(crate) syntax: SyntaxNode,
}
impl ast::AttrsOwner for ForExpr {}
impl ast::LoopBodyOwner for ForExpr {}
impl ForExpr {
    pub fn for_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![for]) }
    pub fn pat(&self) -> Option<Pat> { support::child(&self.syntax) }
    pub fn in_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![in]) }
    pub fn iterable(&self) -> Option<Expr> { support::child(&self.syntax) }
}
/// While loop expression. Includes both regular `while` and `while let` forms.
///
/// ```
/// ❰
///     while bool_cond {
///         42;
///     }
/// ❱
/// ❰
///     while let Pattern(foo) = bar {
///         bar += 1;
///     }
/// ❱
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/expressions/loop-expr.html#predicate-loops)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WhileExpr {
    pub(crate) syntax: SyntaxNode,
}
impl ast::AttrsOwner for WhileExpr {}
impl ast::LoopBodyOwner for WhileExpr {}
impl WhileExpr {
    pub fn while_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![while]) }
    pub fn condition(&self) -> Option<Condition> { support::child(&self.syntax) }
}
/// Continue expression.
///
/// ```
/// while bool_cond {
///     ❰ continue ❱;
/// }
///
/// 'outer: loop {
///     loop {
///         ❰ continue 'outer ❱;
///     }
/// }
///
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/expressions/loop-expr.html#continue-expressions)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ContinueExpr {
    pub(crate) syntax: SyntaxNode,
}
impl ast::AttrsOwner for ContinueExpr {}
impl ContinueExpr {
    pub fn continue_token(&self) -> Option<SyntaxToken> {
        support::token(&self.syntax, T![continue])
    }
    pub fn lifetime_token(&self) -> Option<SyntaxToken> {
        support::token(&self.syntax, T![lifetime])
    }
}
/// Break expression.
///
/// ```
/// while bool_cond {
///     ❰ break ❱;
/// }
/// 'outer: loop {
///     for foo in bar {
///         ❰ break 'outer ❱;
///     }
/// }
/// 'outer: loop {
///     loop {
///         ❰ break 'outer 42 ❱;
///     }
/// }
/// ```
///
/// [Refernce](https://doc.rust-lang.org/reference/expressions/loop-expr.html#break-expressions)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BreakExpr {
    pub(crate) syntax: SyntaxNode,
}
impl ast::AttrsOwner for BreakExpr {}
impl BreakExpr {
    pub fn break_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![break]) }
    pub fn lifetime_token(&self) -> Option<SyntaxToken> {
        support::token(&self.syntax, T![lifetime])
    }
    pub fn expr(&self) -> Option<Expr> { support::child(&self.syntax) }
}
/// Label.
///
/// ```
/// ❰ 'outer: ❱ loop {}
///
/// let foo = ❰ 'bar: ❱ loop {}
///
/// ❰ 'baz: ❱ {
///     break 'baz;
/// }
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/expressions/loop-expr.html?highlight=label#loop-labels)
/// [Labels for blocks RFC](https://github.com/rust-lang/rfcs/blob/master/text/2046-label-break-value.md)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Label {
    pub(crate) syntax: SyntaxNode,
}
impl Label {
    pub fn lifetime_token(&self) -> Option<SyntaxToken> {
        support::token(&self.syntax, T![lifetime])
    }
}
/// Block expression. Includes unsafe blocks and block labels.
///
/// ```
///     let foo = ❰
///         {
///             #![inner_attr]
///             ❰ { } ❱
///
///             ❰ 'label: { break 'label } ❱
///         }
///     ❱;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/expressions/block-expr.html)
/// [Labels for blocks RFC](https://github.com/rust-lang/rfcs/blob/master/text/2046-label-break-value.md)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BlockExpr {
    pub(crate) syntax: SyntaxNode,
}
impl ast::AttrsOwner for BlockExpr {}
impl ast::ModuleItemOwner for BlockExpr {}
impl BlockExpr {
    pub fn label(&self) -> Option<Label> { support::child(&self.syntax) }
    pub fn l_curly_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T!['{']) }
    pub fn statements(&self) -> AstChildren<Stmt> { support::children(&self.syntax) }
    pub fn expr(&self) -> Option<Expr> { support::child(&self.syntax) }
    pub fn r_curly_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T!['}']) }
}
/// Return expression.
///
/// ```
/// || ❰ return 42 ❱;
///
/// fn bar() {
///     ❰ return ❱;
/// }
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/expressions/return-expr.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ReturnExpr {
    pub(crate) syntax: SyntaxNode,
}
impl ast::AttrsOwner for ReturnExpr {}
impl ReturnExpr {
    pub fn expr(&self) -> Option<Expr> { support::child(&self.syntax) }
}
/// Call expression (not to be confused with method call expression, it is
/// a separate ast node).
///
/// ```
/// ❰ foo() ❱;
/// ❰ &str::len("bar") ❱;
/// ❰ <&str as PartialEq<&str>>::eq(&"", &"") ❱;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/expressions/call-expr.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CallExpr {
    pub(crate) syntax: SyntaxNode,
}
impl ast::ArgListOwner for CallExpr {}
impl CallExpr {
    pub fn expr(&self) -> Option<Expr> { support::child(&self.syntax) }
}
/// Method call expression.
///
/// ```
/// ❰ receiver_expr.method() ❱;
/// ❰ receiver_expr.method::<T>(42, true) ❱;
///
/// ❰ ❰ ❰ foo.bar() ❱ .baz() ❱ .bruh() ❱;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/expressions/method-call-expr.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MethodCallExpr {
    pub(crate) syntax: SyntaxNode,
}
impl ast::AttrsOwner for MethodCallExpr {}
impl ast::ArgListOwner for MethodCallExpr {}
impl MethodCallExpr {
    pub fn expr(&self) -> Option<Expr> { support::child(&self.syntax) }
    pub fn dot_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![.]) }
    pub fn name_ref(&self) -> Option<NameRef> { support::child(&self.syntax) }
    pub fn type_arg_list(&self) -> Option<TypeArgList> { support::child(&self.syntax) }
}
/// Index expression a.k.a. subscript operator call.
///
/// ```
/// ❰ foo[42] ❱;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/expressions/array-expr.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IndexExpr {
    pub(crate) syntax: SyntaxNode,
}
impl ast::AttrsOwner for IndexExpr {}
impl IndexExpr {
    pub fn l_brack_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T!['[']) }
    pub fn r_brack_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![']']) }
}
/// Field access expression.
///
/// ```
/// ❰ expr.bar ❱;
///
/// ❰ ❰ ❰ foo.bar ❱ .baz ❱ .bruh ❱;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/expressions/field-expr.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FieldExpr {
    pub(crate) syntax: SyntaxNode,
}
impl ast::AttrsOwner for FieldExpr {}
impl FieldExpr {
    pub fn expr(&self) -> Option<Expr> { support::child(&self.syntax) }
    pub fn dot_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![.]) }
    pub fn name_ref(&self) -> Option<NameRef> { support::child(&self.syntax) }
}
/// Await operator call expression.
///
/// ```
/// ❰ expr.await ❱;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/expressions/await-expr.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AwaitExpr {
    pub(crate) syntax: SyntaxNode,
}
impl ast::AttrsOwner for AwaitExpr {}
impl AwaitExpr {
    pub fn expr(&self) -> Option<Expr> { support::child(&self.syntax) }
    pub fn dot_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![.]) }
    pub fn await_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![await]) }
}
/// The question mark operator call.
///
/// ```
/// ❰ expr? ❱;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/expressions/operator-expr.html#the-question-mark-operator)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TryExpr {
    pub(crate) syntax: SyntaxNode,
}
impl ast::AttrsOwner for TryExpr {}
impl TryExpr {
    pub fn expr(&self) -> Option<Expr> { support::child(&self.syntax) }
    pub fn question_mark_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![?]) }
}
/// Type cast expression.
///
/// ```
/// ❰ expr as T ❱;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/expressions/operator-expr.html#type-cast-expressions)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CastExpr {
    pub(crate) syntax: SyntaxNode,
}
impl ast::AttrsOwner for CastExpr {}
impl CastExpr {
    pub fn expr(&self) -> Option<Expr> { support::child(&self.syntax) }
    pub fn as_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![as]) }
    pub fn type_ref(&self) -> Option<TypeRef> { support::child(&self.syntax) }
}
/// Borrow operator call.
///
/// ```
/// ❰ &foo ❱;
/// ❰ &mut bar ❱;
/// ❰ &raw const bar ❱;
/// ❰ &raw mut bar ❱;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/expressions/operator-expr.html#borrow-operators)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RefExpr {
    pub(crate) syntax: SyntaxNode,
}
impl ast::AttrsOwner for RefExpr {}
impl RefExpr {
    pub fn amp_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![&]) }
    pub fn raw_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![raw]) }
    pub fn mut_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![mut]) }
    pub fn const_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![const]) }
    pub fn expr(&self) -> Option<Expr> { support::child(&self.syntax) }
}
/// Prefix operator call. This is either `!` or `*` or `-`.
///
/// ```
/// ❰ !foo ❱;
/// ❰ *bar ❱;
/// ❰ -42 ❱;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/expressions/operator-expr.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PrefixExpr {
    pub(crate) syntax: SyntaxNode,
}
impl ast::AttrsOwner for PrefixExpr {}
impl PrefixExpr {
    pub fn expr(&self) -> Option<Expr> { support::child(&self.syntax) }
}
/// Box operator call.
///
/// ```
/// ❰ box 42 ❱;
/// ```
///
/// [RFC](https://github.com/rust-lang/rfcs/blob/0806be4f282144cfcd55b1d20284b43f87cbe1c6/text/0809-box-and-in-for-stdlib.md)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BoxExpr {
    pub(crate) syntax: SyntaxNode,
}
impl ast::AttrsOwner for BoxExpr {}
impl BoxExpr {
    pub fn box_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![box]) }
    pub fn expr(&self) -> Option<Expr> { support::child(&self.syntax) }
}
/// Range operator call.
///
/// ```
/// ❰ 0..42 ❱;
/// ❰ ..42 ❱;
/// ❰ 0.. ❱;
/// ❰ .. ❱;
/// ❰ 0..=42 ❱;
/// ❰ ..=42 ❱;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/expressions/range-expr.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RangeExpr {
    pub(crate) syntax: SyntaxNode,
}
impl ast::AttrsOwner for RangeExpr {}
impl RangeExpr {}
/// Binary operator call.
/// Includes all arithmetic, logic, bitwise and assignment operators.
///
/// ```
/// ❰ 2 + ❰ 2 * 2 ❱ ❱;
/// ❰ ❰ true && false ❱ || true ❱;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/expressions/operator-expr.html#arithmetic-and-logical-binary-operators)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BinExpr {
    pub(crate) syntax: SyntaxNode,
}
impl ast::AttrsOwner for BinExpr {}
impl BinExpr {}
/// [Raw] string, [raw] byte string, char, byte, integer, float or bool literal.
///
/// ```
/// ❰ "str" ❱;
/// ❰ br##"raw byte str"## ❱;
/// ❰ 'c' ❱;
/// ❰ b'c' ❱;
/// ❰ 42 ❱;
/// ❰ 1e9 ❱;
/// ❰ true ❱;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/expressions/literal-expr.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Literal {
    pub(crate) syntax: SyntaxNode,
}
impl Literal {}
/// Match expression.
///
/// ```
/// ❰
///     match expr {
///         Pat1 => {}
///         Pat2(_) => 42,
///     }
/// ❱
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/expressions/match-expr.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MatchExpr {
    pub(crate) syntax: SyntaxNode,
}
impl ast::AttrsOwner for MatchExpr {}
impl MatchExpr {
    pub fn match_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![match]) }
    pub fn expr(&self) -> Option<Expr> { support::child(&self.syntax) }
    pub fn match_arm_list(&self) -> Option<MatchArmList> { support::child(&self.syntax) }
}
/// Match arm list part of match expression. Includes its inner attributes.
///
/// ```
/// match expr
/// ❰
///     {
///         #![inner_attr]
///         Pat1 => {}
///         Pat2(_) => 42,
///     }
/// ❱
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/expressions/match-expr.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MatchArmList {
    pub(crate) syntax: SyntaxNode,
}
impl ast::AttrsOwner for MatchArmList {}
impl MatchArmList {
    pub fn l_curly_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T!['{']) }
    pub fn arms(&self) -> AstChildren<MatchArm> { support::children(&self.syntax) }
    pub fn r_curly_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T!['}']) }
}
/// Match arm.
/// Note: record struct literals are not valid as target match expression
/// due to ambiguity.
/// ```
/// match expr {
///     ❰ #[attr] Pattern(it) if bool_cond => it ❱,
/// }
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/expressions/match-expr.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MatchArm {
    pub(crate) syntax: SyntaxNode,
}
impl ast::AttrsOwner for MatchArm {}
impl MatchArm {
    pub fn pat(&self) -> Option<Pat> { support::child(&self.syntax) }
    pub fn guard(&self) -> Option<MatchGuard> { support::child(&self.syntax) }
    pub fn fat_arrow_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![=>]) }
    pub fn expr(&self) -> Option<Expr> { support::child(&self.syntax) }
}
/// Match guard.
///
/// ```
/// match expr {
///     Pattern(it) ❰ if bool_cond ❱ => it,
/// }
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/expressions/match-expr.html#match-guards)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MatchGuard {
    pub(crate) syntax: SyntaxNode,
}
impl MatchGuard {
    pub fn if_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![if]) }
    pub fn expr(&self) -> Option<Expr> { support::child(&self.syntax) }
}
/// Record literal expression. The same syntax is used for structs,
/// unions and record enum variants.
///
/// ```
/// ❰
///     foo::Bar {
///         #![inner_attr]
///         baz: 42,
///         bruh: true,
///         ..spread
///     }
/// ❱
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/expressions/struct-expr.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RecordLit {
    pub(crate) syntax: SyntaxNode,
}
impl RecordLit {
    pub fn path(&self) -> Option<Path> { support::child(&self.syntax) }
    pub fn record_field_list(&self) -> Option<RecordFieldList> { support::child(&self.syntax) }
}
/// Record field list including enclosing curly braces.
///
/// foo::Bar ❰
///     {
///         baz: 42,
///         ..spread
///     }
/// ❱
///
/// [Reference](https://doc.rust-lang.org/reference/expressions/struct-expr.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RecordFieldList {
    pub(crate) syntax: SyntaxNode,
}
impl RecordFieldList {
    pub fn l_curly_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T!['{']) }
    pub fn fields(&self) -> AstChildren<RecordField> { support::children(&self.syntax) }
    pub fn dotdot_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![..]) }
    pub fn spread(&self) -> Option<Expr> { support::child(&self.syntax) }
    pub fn r_curly_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T!['}']) }
}
/// Record field.
///
/// ```
/// foo::Bar {
///     ❰ #[attr] baz: 42 ❱
/// }
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/expressions/struct-expr.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RecordField {
    pub(crate) syntax: SyntaxNode,
}
impl ast::AttrsOwner for RecordField {}
impl RecordField {
    pub fn name_ref(&self) -> Option<NameRef> { support::child(&self.syntax) }
    pub fn colon_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![:]) }
    pub fn expr(&self) -> Option<Expr> { support::child(&self.syntax) }
}
/// Disjunction of patterns.
///
/// ```
/// let ❰ Foo(it) | Bar(it) | Baz(it) ❱ = bruh;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/patterns.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OrPat {
    pub(crate) syntax: SyntaxNode,
}
impl OrPat {
    pub fn pats(&self) -> AstChildren<Pat> { support::children(&self.syntax) }
}
/// Parenthesized pattern.
/// Note: parens are only used for grouping, this is not a tuple pattern.
///
/// ```
/// if let ❰ &(0..=42) ❱ = foo {}
/// ```
///
/// https://doc.rust-lang.org/reference/patterns.html#grouped-patterns
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParenPat {
    pub(crate) syntax: SyntaxNode,
}
impl ParenPat {
    pub fn l_paren_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T!['(']) }
    pub fn pat(&self) -> Option<Pat> { support::child(&self.syntax) }
    pub fn r_paren_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![')']) }
}
/// Reference pattern.
/// Note: this has nothing to do with `ref` keyword, the latter is used in bind patterns.
///
/// ```
/// let ❰ &mut foo ❱ = bar;
///
/// let ❰ & ❰ &mut ❰ &_ ❱ ❱ ❱ = baz;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/patterns.html#reference-patterns)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RefPat {
    pub(crate) syntax: SyntaxNode,
}
impl RefPat {
    pub fn amp_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![&]) }
    pub fn mut_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![mut]) }
    pub fn pat(&self) -> Option<Pat> { support::child(&self.syntax) }
}
/// Box pattern.
///
/// ```
/// let ❰ box foo ❱ = box 42;
/// ```
///
/// [Unstable book](https://doc.rust-lang.org/unstable-book/language-features/box-patterns.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BoxPat {
    pub(crate) syntax: SyntaxNode,
}
impl BoxPat {
    pub fn box_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![box]) }
    pub fn pat(&self) -> Option<Pat> { support::child(&self.syntax) }
}
/// Bind pattern.
///
/// ```
/// match foo {
///     Some(❰ ref mut bar ❱) => {}
///     ❰ baz @ None ❱ => {}
/// }
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/patterns.html#identifier-patterns)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BindPat {
    pub(crate) syntax: SyntaxNode,
}
impl ast::AttrsOwner for BindPat {}
impl ast::NameOwner for BindPat {}
impl BindPat {
    pub fn ref_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![ref]) }
    pub fn mut_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![mut]) }
    pub fn at_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![@]) }
    pub fn pat(&self) -> Option<Pat> { support::child(&self.syntax) }
}
/// Placeholder pattern a.k.a. the wildcard pattern or the underscore.
///
/// ```
/// let ❰ _ ❱ = foo;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/patterns.html#wildcard-pattern)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PlaceholderPat {
    pub(crate) syntax: SyntaxNode,
}
impl PlaceholderPat {
    pub fn underscore_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![_]) }
}
/// Rest-of-the record/tuple pattern.
/// Note: this is not the unbonded range pattern (even more: it doesn't exist).
///
/// ```
/// let Foo { bar, ❰ .. ❱ } = baz;
/// let (❰ .. ❱, bruh) = (42, 24, 42);
/// let Bruuh(❰ .. ❱) = bruuuh;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/patterns.html#struct-patterns)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DotDotPat {
    pub(crate) syntax: SyntaxNode,
}
impl DotDotPat {
    pub fn dotdot_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![..]) }
}
/// Path pattern.
/// Doesn't include the underscore pattern (it is a special case, namely `PlaceholderPat`).
///
/// ```
/// let ❰ foo::bar::Baz ❱ { .. } = bruh;
/// if let ❰ CONST ❱ = 42 {}
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/patterns.html#path-patterns)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PathPat {
    pub(crate) syntax: SyntaxNode,
}
impl PathPat {
    pub fn path(&self) -> Option<Path> { support::child(&self.syntax) }
}
/// Slice pattern.
///
/// ```
/// let ❰ [foo, bar, baz] ❱ = [1, 2, 3];
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/patterns.html#slice-patterns)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SlicePat {
    pub(crate) syntax: SyntaxNode,
}
impl SlicePat {
    pub fn l_brack_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T!['[']) }
    pub fn args(&self) -> AstChildren<Pat> { support::children(&self.syntax) }
    pub fn r_brack_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![']']) }
}
/// Range pattern.
///
/// ```
/// match foo {
///     ❰ 0..42 ❱ => {}
///     ❰ 0..=42 ❱ => {}
/// }
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/patterns.html#range-patterns)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RangePat {
    pub(crate) syntax: SyntaxNode,
}
impl RangePat {}
/// Literal pattern.
/// Includes only bool, number, char, and string literals.
///
/// ```
/// match foo {
///     Number(❰ 42 ❱) => {}
///     String(❰ "42" ❱) => {}
///     Bool(❰ true ❱) => {}
/// }
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/patterns.html#literal-patterns)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LiteralPat {
    pub(crate) syntax: SyntaxNode,
}
impl LiteralPat {
    pub fn literal(&self) -> Option<Literal> { support::child(&self.syntax) }
}
/// Macro invocation in pattern position.
///
/// ```
/// let ❰ foo!(my custom syntax) ❱ = baz;
///
/// ```
/// [Reference](https://doc.rust-lang.org/reference/macros.html#macro-invocation)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MacroPat {
    pub(crate) syntax: SyntaxNode,
}
impl MacroPat {
    pub fn macro_call(&self) -> Option<MacroCall> { support::child(&self.syntax) }
}
/// Record literal pattern.
///
/// ```
/// let ❰ foo::Bar { baz, .. } ❱ = bruh;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/patterns.html#struct-patterns)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RecordPat {
    pub(crate) syntax: SyntaxNode,
}
impl RecordPat {
    pub fn record_field_pat_list(&self) -> Option<RecordFieldPatList> {
        support::child(&self.syntax)
    }
    pub fn path(&self) -> Option<Path> { support::child(&self.syntax) }
}
/// Record literal's field patterns list including enclosing curly braces.
///
/// ```
/// let foo::Bar ❰ { baz, bind @ bruh, .. } ❱ = bruuh;
/// ``
///
/// [Reference](https://doc.rust-lang.org/reference/patterns.html#struct-patterns)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RecordFieldPatList {
    pub(crate) syntax: SyntaxNode,
}
impl RecordFieldPatList {
    pub fn l_curly_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T!['{']) }
    pub fn pats(&self) -> AstChildren<RecordInnerPat> { support::children(&self.syntax) }
    pub fn record_field_pats(&self) -> AstChildren<RecordFieldPat> {
        support::children(&self.syntax)
    }
    pub fn bind_pats(&self) -> AstChildren<BindPat> { support::children(&self.syntax) }
    pub fn dotdot_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![..]) }
    pub fn r_curly_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T!['}']) }
}
/// Record literal's field pattern.
/// Note: record literal can also match tuple structs.
///
/// ```
/// let Foo { ❰ bar: _ ❱ } = baz;
/// let TupleStruct { ❰ 0: _ ❱ } = bruh;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/patterns.html#struct-patterns)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RecordFieldPat {
    pub(crate) syntax: SyntaxNode,
}
impl ast::AttrsOwner for RecordFieldPat {}
impl RecordFieldPat {
    pub fn name_ref(&self) -> Option<NameRef> { support::child(&self.syntax) }
    pub fn colon_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![:]) }
    pub fn pat(&self) -> Option<Pat> { support::child(&self.syntax) }
}
/// Tuple struct literal pattern.
///
/// ```
/// let ❰ foo::Bar(baz, bruh) ❱ = bruuh;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/patterns.html#tuple-struct-patterns)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TupleStructPat {
    pub(crate) syntax: SyntaxNode,
}
impl TupleStructPat {
    pub fn path(&self) -> Option<Path> { support::child(&self.syntax) }
    pub fn l_paren_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T!['(']) }
    pub fn args(&self) -> AstChildren<Pat> { support::children(&self.syntax) }
    pub fn r_paren_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![')']) }
}
/// Tuple pattern.
/// Note: this doesn't include tuple structs (see `TupleStructPat`)
///
/// ```
/// let ❰ (foo, bar, .., baz) ❱ = bruh;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/patterns.html#tuple-patterns)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TuplePat {
    pub(crate) syntax: SyntaxNode,
}
impl TuplePat {
    pub fn l_paren_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T!['(']) }
    pub fn args(&self) -> AstChildren<Pat> { support::children(&self.syntax) }
    pub fn r_paren_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![')']) }
}
/// Visibility.
///
/// ```
/// ❰ pub mod ❱ foo;
/// ❰ pub(crate) ❱ struct Bar;
/// ❰ pub(self) ❱ enum Baz {}
/// ❰ pub(super) ❱ fn bruh() {}
/// ❰ pub(in bruuh::bruuuh) ❱ type T = u64;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/visibility-and-privacy.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Visibility {
    pub(crate) syntax: SyntaxNode,
}
impl Visibility {
    pub fn pub_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![pub]) }
    pub fn super_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![super]) }
    pub fn self_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![self]) }
    pub fn crate_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![crate]) }
}
/// Single identifier.
/// Note(@matklad): `Name` is for things that install a new name into the scope,
/// `NameRef` is a usage of a name. Most of the time, this definition/reference
/// distinction can be determined purely syntactically, ie in
/// ```
/// fn foo() { foo() }
/// ```
/// the first foo is `Name`, the second one is `NameRef`.
/// The notable exception are patterns, where in
/// ``
/// let x = 92
/// ```
/// `x` can be semantically either a name or a name ref, depeding on
/// wether there's an `x` constant in scope.
/// We use `Name` for patterns, and disambiguate semantically (see `NameClass` in ide_db).
///
/// ```
/// let ❰ foo ❱ = bar;
/// struct ❰ Baz ❱;
/// fn ❰ bruh ❱() {}
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/identifiers.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Name {
    pub(crate) syntax: SyntaxNode,
}
impl Name {
    pub fn ident_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![ident]) }
}
/// Reference to a name.
/// See the explanation on the difference between `Name` and `NameRef`
/// in `Name` ast node docs.
///
/// ```
/// let foo = ❰ bar ❱(❰ Baz(❰ bruh ❱) ❱;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/identifiers.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NameRef {
    pub(crate) syntax: SyntaxNode,
}
impl NameRef {}
/// Macro call.
/// Includes all of its attributes and doc comments.
///
/// ```
/// ❰
///     /// Docs
///     #[attr]
///     macro_rules! foo {   // macro rules is also a macro call
///         ($bar: tt) => {}
///     }
/// ❱
///
/// // semicolon is a part of `MacroCall` when it is used in item positions
/// ❰ foo!(); ❱
///
/// fn main() {
///     ❰ foo!() ❱; // macro call in expression positions doesn't include the semi
/// }
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/macros.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MacroCall {
    pub(crate) syntax: SyntaxNode,
}
impl ast::NameOwner for MacroCall {}
impl ast::AttrsOwner for MacroCall {}
impl ast::DocCommentsOwner for MacroCall {}
impl MacroCall {
    pub fn path(&self) -> Option<Path> { support::child(&self.syntax) }
    pub fn excl_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![!]) }
    pub fn token_tree(&self) -> Option<TokenTree> { support::child(&self.syntax) }
    pub fn semicolon_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![;]) }
}
/// Attribute.
///
/// ```
/// ❰ #![inner_attr] ❱
///
/// ❰ #[attr] ❱
/// ❰ #[foo = "bar"] ❱
/// ❰ #[baz(bruh::bruuh = "42")] ❱
/// struct Foo;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/attributes.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Attr {
    pub(crate) syntax: SyntaxNode,
}
impl Attr {
    pub fn pound_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![#]) }
    pub fn excl_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![!]) }
    pub fn l_brack_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T!['[']) }
    pub fn path(&self) -> Option<Path> { support::child(&self.syntax) }
    pub fn eq_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![=]) }
    pub fn input(&self) -> Option<AttrInput> { support::child(&self.syntax) }
    pub fn r_brack_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![']']) }
}
/// Stores a list of lexer tokens and other `TokenTree`s.
/// It appears in attributes, macro_rules and macro call (foo!)
///
/// ```
/// macro_call! ❰ { my syntax here } ❱;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/macros.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TokenTree {
    pub(crate) syntax: SyntaxNode,
}
impl TokenTree {}
/// Generic lifetime, type and constants parameters list **declaration**.
///
/// ```
/// fn foo❰ <'a, 'b, T, U, const BAR: u64> ❱() {}
///
/// struct Baz❰ <T> ❱(T);
///
/// impl❰ <T> ❱ Bruh<T> {}
///
/// type Bruuh = for❰ <'a> ❱ fn(&'a str) -> &'a str;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/items/generics.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypeParamList {
    pub(crate) syntax: SyntaxNode,
}
impl TypeParamList {
    pub fn l_angle_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![<]) }
    pub fn generic_params(&self) -> AstChildren<GenericParam> { support::children(&self.syntax) }
    pub fn type_params(&self) -> AstChildren<TypeParam> { support::children(&self.syntax) }
    pub fn lifetime_params(&self) -> AstChildren<LifetimeParam> { support::children(&self.syntax) }
    pub fn const_params(&self) -> AstChildren<ConstParam> { support::children(&self.syntax) }
    pub fn r_angle_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![>]) }
}
/// Single type parameter **declaration**.
///
/// ```
/// fn foo<❰ K ❱, ❰ I ❱, ❰ E: Debug ❱, ❰ V = DefaultType ❱>() {}
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/items/generics.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypeParam {
    pub(crate) syntax: SyntaxNode,
}
impl ast::NameOwner for TypeParam {}
impl ast::AttrsOwner for TypeParam {}
impl ast::TypeBoundsOwner for TypeParam {}
impl TypeParam {
    pub fn eq_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![=]) }
    pub fn default_type(&self) -> Option<TypeRef> { support::child(&self.syntax) }
}
/// Const generic parameter **declaration**.
/// ```
/// fn foo<T, U, ❰ const BAR: usize ❱, ❰ const BAZ: bool ❱>() {}
/// ```
///
/// [RFC](https://github.com/rust-lang/rfcs/blob/master/text/2000-const-generics.md#declaring-a-const-parameter)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConstParam {
    pub(crate) syntax: SyntaxNode,
}
impl ast::NameOwner for ConstParam {}
impl ast::AttrsOwner for ConstParam {}
impl ast::TypeAscriptionOwner for ConstParam {}
impl ConstParam {
    pub fn eq_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![=]) }
    pub fn default_val(&self) -> Option<Expr> { support::child(&self.syntax) }
}
/// Lifetime parameter **declaration**.
///
/// ```
/// fn foo<❰ 'a ❱, ❰ 'b ❱, V, G, D>(bar: &'a str, baz: &'b mut str) {}
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/items/generics.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LifetimeParam {
    pub(crate) syntax: SyntaxNode,
}
impl ast::AttrsOwner for LifetimeParam {}
impl LifetimeParam {
    pub fn lifetime_token(&self) -> Option<SyntaxToken> {
        support::token(&self.syntax, T![lifetime])
    }
}
/// Type bound declaration clause.
///
/// ```
/// fn foo<T: ❰ ?Sized ❱ + ❰ Debug ❱>() {}
///
/// trait Bar<T>
/// where
///     T: ❰ Send ❱ + ❰ Sync ❱
/// {
///     type Baz: ❰ !Sync ❱ + ❰ Debug ❱ + ❰ ?const Add ❱;
/// }
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/trait-bounds.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypeBound {
    pub(crate) syntax: SyntaxNode,
}
impl TypeBound {
    pub fn lifetime_token(&self) -> Option<SyntaxToken> {
        support::token(&self.syntax, T![lifetime])
    }
    pub fn const_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![const]) }
    pub fn type_ref(&self) -> Option<TypeRef> { support::child(&self.syntax) }
}
/// Type bounds list.
///
/// ```
///
/// fn foo<T: ❰ ?Sized + Debug ❱>() {}
///
/// trait Bar<T>
/// where
///     T: ❰ Send + Sync ❱
/// {
///     type Baz: ❰ !Sync + Debug ❱;
/// }
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/trait-bounds.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypeBoundList {
    pub(crate) syntax: SyntaxNode,
}
impl TypeBoundList {
    pub fn bounds(&self) -> AstChildren<TypeBound> { support::children(&self.syntax) }
}
/// Single where predicate.
///
/// ```
/// trait Foo<'a, 'b, T>
/// where
///     ❰ 'a: 'b ❱,
///     ❰ T: IntoIterator ❱,
///     ❰ for<'c> <T as IntoIterator>::Item: Bar<'c> ❱
/// {}
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/items/generics.html#where-clauses)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WherePred {
    pub(crate) syntax: SyntaxNode,
}
impl ast::TypeBoundsOwner for WherePred {}
impl WherePred {
    pub fn for_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![for]) }
    pub fn type_param_list(&self) -> Option<TypeParamList> { support::child(&self.syntax) }
    pub fn lifetime_token(&self) -> Option<SyntaxToken> {
        support::token(&self.syntax, T![lifetime])
    }
    pub fn type_ref(&self) -> Option<TypeRef> { support::child(&self.syntax) }
}
/// Where clause.
///
/// ```
/// trait Foo<'a, T> ❰ where 'a: 'static, T: Debug ❱ {}
///
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/items/generics.html#where-clauses)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WhereClause {
    pub(crate) syntax: SyntaxNode,
}
impl WhereClause {
    pub fn where_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![where]) }
    pub fn predicates(&self) -> AstChildren<WherePred> { support::children(&self.syntax) }
}
/// Abi declaration.
/// Note: the abi string is optional.
///
/// ```
/// ❰ extern "C" ❱ {
///     fn foo() {}
/// }
///
/// type Bar = ❰ extern ❱ fn() -> u32;
///
/// type Baz = ❰ extern r#"stdcall"# ❱ fn() -> bool;
/// ```
///
/// - [Extern blocks reference](https://doc.rust-lang.org/reference/items/external-blocks.html)
/// - [FFI function pointers reference](https://doc.rust-lang.org/reference/items/functions.html#functions)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Abi {
    pub(crate) syntax: SyntaxNode,
}
impl Abi {}
/// Expression statement.
///
/// ```
/// ❰ 42; ❱
/// ❰ foo(); ❱
/// ❰ (); ❱
/// ❰ {}; ❱
///
/// // constructions with trailing curly brace can omit the semicolon
/// // but only when there are satements immediately after them (this is important!)
/// ❰ if bool_cond { } ❱
/// ❰ loop {} ❱
/// ❰ somestatment; ❱
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/statements.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExprStmt {
    pub(crate) syntax: SyntaxNode,
}
impl ast::AttrsOwner for ExprStmt {}
impl ExprStmt {
    pub fn expr(&self) -> Option<Expr> { support::child(&self.syntax) }
    pub fn semicolon_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![;]) }
}
/// Let statement.
///
/// ```
/// ❰ #[attr] let foo; ❱
/// ❰ let bar: u64; ❱
/// ❰ let baz = 42; ❱
/// ❰ let bruh: bool = true; ❱
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/statements.html#let-statements)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LetStmt {
    pub(crate) syntax: SyntaxNode,
}
impl ast::AttrsOwner for LetStmt {}
impl ast::TypeAscriptionOwner for LetStmt {}
impl LetStmt {
    pub fn let_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![let]) }
    pub fn pat(&self) -> Option<Pat> { support::child(&self.syntax) }
    pub fn eq_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![=]) }
    pub fn initializer(&self) -> Option<Expr> { support::child(&self.syntax) }
    pub fn semicolon_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![;]) }
}
/// Condition of `if` or `while` expression.
///
/// ```
/// if ❰ true ❱ {}
/// if ❰ let Pat(foo) = bar ❱ {}
///
/// while ❰ true ❱ {}
/// while ❰ let Pat(baz) = bruh ❱ {}
/// ```
///
/// [If expression reference](https://doc.rust-lang.org/reference/expressions/if-expr.html)
/// [While expression reference](https://doc.rust-lang.org/reference/expressions/loop-expr.html#predicate-loops)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Condition {
    pub(crate) syntax: SyntaxNode,
}
impl Condition {
    pub fn let_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![let]) }
    pub fn pat(&self) -> Option<Pat> { support::child(&self.syntax) }
    pub fn eq_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![=]) }
    pub fn expr(&self) -> Option<Expr> { support::child(&self.syntax) }
}
/// Parameter list **declaration**.
///
/// ```
/// fn foo❰ (a: u32, b: bool) ❱ -> u32 {}
/// let bar = ❰ |a, b| ❱ {};
///
/// impl Baz {
///     fn bruh❰ (&self, a: u32) ❱ {}
/// }
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/items/functions.html)ocs to codegen script
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParamList {
    pub(crate) syntax: SyntaxNode,
}
impl ParamList {
    pub fn l_paren_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T!['(']) }
    pub fn self_param(&self) -> Option<SelfParam> { support::child(&self.syntax) }
    pub fn params(&self) -> AstChildren<Param> { support::children(&self.syntax) }
    pub fn r_paren_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![')']) }
}
/// Self parameter **declaration**.
///
/// ```
/// impl Bruh {
///     fn foo(❰ self ❱) {}
///     fn bar(❰ &self ❱) {}
///     fn baz(❰ &mut self ❱) {}
///     fn blah<'a>(❰ &'a self ❱) {}
///     fn blin(❰ self: Box<Self> ❱) {}
/// }
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/items/functions.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SelfParam {
    pub(crate) syntax: SyntaxNode,
}
impl ast::TypeAscriptionOwner for SelfParam {}
impl ast::AttrsOwner for SelfParam {}
impl SelfParam {
    pub fn amp_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![&]) }
    pub fn mut_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![mut]) }
    pub fn lifetime_token(&self) -> Option<SyntaxToken> {
        support::token(&self.syntax, T![lifetime])
    }
    pub fn self_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![self]) }
}
/// Parameter **declaration**.
///
/// ```
/// fn foo(❰ #[attr] Pat(bar): Pat(u32) ❱, ❰ #[attr] _: bool ❱) {}
///
/// extern "C" {
///     fn bar(❰ baz: u32 ❱, ❰ ... ❱) -> u32;
/// }
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/items/functions.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Param {
    pub(crate) syntax: SyntaxNode,
}
impl ast::TypeAscriptionOwner for Param {}
impl ast::AttrsOwner for Param {}
impl Param {
    pub fn pat(&self) -> Option<Pat> { support::child(&self.syntax) }
    pub fn dotdotdot_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![...]) }
}
/// Use declaration.
///
/// ```
/// ❰ #[attr] pub use foo; ❱
/// ❰ use bar as baz; ❱
/// ❰ use bruh::{self, bruuh}; ❱
/// ❰ use { blin::blen, blah::* };
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/items/use-declarations.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UseItem {
    pub(crate) syntax: SyntaxNode,
}
impl ast::AttrsOwner for UseItem {}
impl ast::VisibilityOwner for UseItem {}
impl UseItem {
    pub fn use_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![use]) }
    pub fn use_tree(&self) -> Option<UseTree> { support::child(&self.syntax) }
}
/// Use tree.
///
/// ```
/// pub use ❰ foo::❰ * ❱ ❱;
/// use ❰ bar as baz ❱;
/// use ❰ bruh::bruuh::{ ❰ self ❱, ❰ blin ❱ } ❱;
/// use ❰ { ❰ blin::blen ❱ } ❱
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/items/use-declarations.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UseTree {
    pub(crate) syntax: SyntaxNode,
}
impl UseTree {
    pub fn path(&self) -> Option<Path> { support::child(&self.syntax) }
    pub fn star_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![*]) }
    pub fn use_tree_list(&self) -> Option<UseTreeList> { support::child(&self.syntax) }
    pub fn alias(&self) -> Option<Alias> { support::child(&self.syntax) }
}
/// Item alias.
/// Note: this is not the type alias.
///
/// ```
/// use foo ❰ as bar ❱;
/// use baz::{bruh ❰ as _ ❱};
/// extern crate bruuh ❰ as blin ❱;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/items/use-declarations.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Alias {
    pub(crate) syntax: SyntaxNode,
}
impl ast::NameOwner for Alias {}
impl Alias {
    pub fn as_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![as]) }
}
/// Sublist of use trees.
///
/// ```
/// use bruh::bruuh::❰ { ❰ self ❱, ❰ blin ❱ } ❱;
/// use ❰ { blin::blen::❰ {} ❱ } ❱
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/items/use-declarations.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UseTreeList {
    pub(crate) syntax: SyntaxNode,
}
impl UseTreeList {
    pub fn l_curly_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T!['{']) }
    pub fn use_trees(&self) -> AstChildren<UseTree> { support::children(&self.syntax) }
    pub fn r_curly_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T!['}']) }
}
/// Extern crate item.
///
/// ```
/// ❰ #[attr] pub extern crate foo; ❱
/// ❰ extern crate self as bar; ❱
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/items/extern-crates.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExternCrateItem {
    pub(crate) syntax: SyntaxNode,
}
impl ast::AttrsOwner for ExternCrateItem {}
impl ast::VisibilityOwner for ExternCrateItem {}
impl ExternCrateItem {
    pub fn extern_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![extern]) }
    pub fn crate_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![crate]) }
    pub fn name_ref(&self) -> Option<NameRef> { support::child(&self.syntax) }
    pub fn alias(&self) -> Option<Alias> { support::child(&self.syntax) }
}
/// Call site arguments list.
///
/// ```
/// foo::<T, U>❰ (42, true) ❱;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/expressions/call-expr.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ArgList {
    pub(crate) syntax: SyntaxNode,
}
impl ArgList {
    pub fn l_paren_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T!['(']) }
    pub fn args(&self) -> AstChildren<Expr> { support::children(&self.syntax) }
    pub fn r_paren_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![')']) }
}
/// Path to a symbol. Includes single identifier names and elaborate paths with
/// generic parameters.
///
/// ```
/// (0..10).❰ ❰ collect ❱ ::<Vec<_>> ❱();
/// ❰ ❰ ❰ Vec ❱ ::<u8> ❱ ::with_capacity ❱(1024);
/// ❰ ❰ <❰ Foo ❱ as ❰ ❰ bar ❱ ::Bar ❱> ❱ ::baz ❱();
/// ❰ ❰ <❰ bruh ❱> ❱ ::bruuh ❱();
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/paths.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Path {
    pub(crate) syntax: SyntaxNode,
}
impl Path {
    pub fn segment(&self) -> Option<PathSegment> { support::child(&self.syntax) }
    pub fn coloncolon_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![::]) }
    pub fn qualifier(&self) -> Option<Path> { support::child(&self.syntax) }
}
/// Segment of the path to a symbol.
/// Only path segment of an absolute path holds the `::` token,
/// all other `::` tokens that connect path segments reside under `Path` itself.`
///
/// ```
/// (0..10).❰ collect ❱ :: ❰ <Vec<_>> ❱();
/// ❰ Vec ❱ :: ❰ <u8> ❱ :: ❰ with_capacity ❱(1024);
/// ❰ <❰ Foo ❱ as ❰ bar ❱ :: ❰ Bar ❱> ❱ :: ❰ baz ❱();
/// ❰ <❰ bruh ❱> ❱ :: ❰ bruuh ❱();
///
/// // Note that only in this case `::` token is inlcuded:
/// ❰ ::foo ❱;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/paths.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PathSegment {
    pub(crate) syntax: SyntaxNode,
}
impl PathSegment {
    pub fn coloncolon_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![::]) }
    pub fn crate_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![crate]) }
    pub fn self_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![self]) }
    pub fn super_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![super]) }
    pub fn l_angle_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![<]) }
    pub fn name_ref(&self) -> Option<NameRef> { support::child(&self.syntax) }
    pub fn type_arg_list(&self) -> Option<TypeArgList> { support::child(&self.syntax) }
    pub fn param_list(&self) -> Option<ParamList> { support::child(&self.syntax) }
    pub fn ret_type(&self) -> Option<RetType> { support::child(&self.syntax) }
    pub fn path_type(&self) -> Option<PathType> { support::child(&self.syntax) }
    pub fn r_angle_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![>]) }
}
/// List of type arguments that are passed at generic instantiation site.
///
/// ```
/// type _ = Foo ❰ ::<'a, u64, Item = Bar, 42, {true}> ❱::Bar;
///
/// Vec❰ ::<bool> ❱::();
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/paths.html#paths-in-expressions)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypeArgList {
    pub(crate) syntax: SyntaxNode,
}
impl TypeArgList {
    pub fn coloncolon_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![::]) }
    pub fn l_angle_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![<]) }
    pub fn generic_args(&self) -> AstChildren<GenericArg> { support::children(&self.syntax) }
    pub fn type_args(&self) -> AstChildren<TypeArg> { support::children(&self.syntax) }
    pub fn lifetime_args(&self) -> AstChildren<LifetimeArg> { support::children(&self.syntax) }
    pub fn assoc_type_args(&self) -> AstChildren<AssocTypeArg> { support::children(&self.syntax) }
    pub fn const_args(&self) -> AstChildren<ConstArg> { support::children(&self.syntax) }
    pub fn r_angle_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![>]) }
}
/// Type argument that is passed at generic instantiation site.
///
/// ```
/// type _ = Foo::<'a, ❰ u64 ❱, ❰ bool ❱, Item = Bar, 42>::Baz;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/paths.html#paths-in-expressions)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypeArg {
    pub(crate) syntax: SyntaxNode,
}
impl TypeArg {
    pub fn type_ref(&self) -> Option<TypeRef> { support::child(&self.syntax) }
}
/// Associated type argument that is passed at generic instantiation site.
/// ```
/// type Foo = Bar::<'a, u64, bool, ❰ Item = Baz ❱, 42>::Bruh;
///
/// trait Bruh<T>: Iterator<❰ Item: Debug ❱> {}
/// ```
///
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AssocTypeArg {
    pub(crate) syntax: SyntaxNode,
}
impl ast::TypeBoundsOwner for AssocTypeArg {}
impl AssocTypeArg {
    pub fn name_ref(&self) -> Option<NameRef> { support::child(&self.syntax) }
    pub fn eq_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![=]) }
    pub fn type_ref(&self) -> Option<TypeRef> { support::child(&self.syntax) }
}
/// Lifetime argument that is passed at generic instantiation site.
///
/// ```
/// fn foo<'a>(s: &'a str) {
///     bar::<❰ 'a ❱>(s);
/// }
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/paths.html#paths-in-expressions)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LifetimeArg {
    pub(crate) syntax: SyntaxNode,
}
impl LifetimeArg {
    pub fn lifetime_token(&self) -> Option<SyntaxToken> {
        support::token(&self.syntax, T![lifetime])
    }
}
/// Constant value argument that is passed at generic instantiation site.
///
/// ```
/// foo::<u32, ❰ { true } ❱>();
///
/// bar::<❰ { 2 + 2} ❱>();
/// ```
///
/// [RFC](https://github.com/rust-lang/rfcs/blob/master/text/2000-const-generics.md#declaring-a-const-parameter)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConstArg {
    pub(crate) syntax: SyntaxNode,
}
impl ConstArg {
    pub fn literal(&self) -> Option<Literal> { support::child(&self.syntax) }
    pub fn block_expr(&self) -> Option<BlockExpr> { support::child(&self.syntax) }
}
/// FIXME: (@edwin0cheng) Remove it to use ItemList instead
/// https://github.com/rust-analyzer/rust-analyzer/pull/4083#discussion_r422666243
///
/// [Reference](https://doc.rust-lang.org/reference/macros.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MacroItems {
    pub(crate) syntax: SyntaxNode,
}
impl ast::ModuleItemOwner for MacroItems {}
impl MacroItems {}
/// FIXME: (@edwin0cheng) add some documentation here. As per the writing
/// of this comment this ast node is not used.
///
/// ```
/// // FIXME: example here
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/macros.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MacroStmts {
    pub(crate) syntax: SyntaxNode,
}
impl MacroStmts {
    pub fn statements(&self) -> AstChildren<Stmt> { support::children(&self.syntax) }
    pub fn expr(&self) -> Option<Expr> { support::child(&self.syntax) }
}
/// List of items in an extern block.
///
/// ```
/// extern "C" ❰
///     {
///         fn foo();
///         static var: u32;
///     }
/// ❱
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/items/external-blocks.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExternItemList {
    pub(crate) syntax: SyntaxNode,
}
impl ast::ModuleItemOwner for ExternItemList {}
impl ExternItemList {
    pub fn l_curly_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T!['{']) }
    pub fn extern_items(&self) -> AstChildren<ExternItem> { support::children(&self.syntax) }
    pub fn r_curly_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T!['}']) }
}
/// Extern block.
///
/// ```
/// ❰
///     extern "C" {
///         fn foo();
///     }
/// ❱
///
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/items/external-blocks.html)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExternBlock {
    pub(crate) syntax: SyntaxNode,
}
impl ExternBlock {
    pub fn abi(&self) -> Option<Abi> { support::child(&self.syntax) }
    pub fn extern_item_list(&self) -> Option<ExternItemList> { support::child(&self.syntax) }
}
/// Meta item in an attribute.
///
/// ```
/// #[❰ bar::baz = "42" ❱]
/// #[❰ bruh(bruuh("true")) ❱]
/// struct Foo;
/// ```
///
/// [Reference](https://doc.rust-lang.org/reference/attributes.html?highlight=meta,item#meta-item-attribute-syntax)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MetaItem {
    pub(crate) syntax: SyntaxNode,
}
impl MetaItem {
    pub fn path(&self) -> Option<Path> { support::child(&self.syntax) }
    pub fn eq_token(&self) -> Option<SyntaxToken> { support::token(&self.syntax, T![=]) }
    pub fn attr_input(&self) -> Option<AttrInput> { support::child(&self.syntax) }
    pub fn nested_meta_items(&self) -> AstChildren<MetaItem> { support::children(&self.syntax) }
}
/// Macro 2.0 definition.
/// Their syntax is still WIP by rustc team...
/// ```
/// ❰
///     macro foo { }
/// ❱
/// ```
///
/// [RFC](https://github.com/rust-lang/rfcs/blob/master/text/1584-macros.md)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MacroDef {
    pub(crate) syntax: SyntaxNode,
}
impl MacroDef {
    pub fn name(&self) -> Option<Name> { support::child(&self.syntax) }
    pub fn token_tree(&self) -> Option<TokenTree> { support::child(&self.syntax) }
}
/// Any kind of nominal type definition.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NominalDef {
    StructDef(StructDef),
    EnumDef(EnumDef),
    UnionDef(UnionDef),
}
impl ast::NameOwner for NominalDef {}
impl ast::TypeParamsOwner for NominalDef {}
impl ast::AttrsOwner for NominalDef {}
/// Any kind of **declared** generic parameter
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GenericParam {
    LifetimeParam(LifetimeParam),
    TypeParam(TypeParam),
    ConstParam(ConstParam),
}
/// Any kind of generic argument passed at instantiation site
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GenericArg {
    LifetimeArg(LifetimeArg),
    TypeArg(TypeArg),
    ConstArg(ConstArg),
    AssocTypeArg(AssocTypeArg),
}
/// Any kind of construct valid in type context
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeRef {
    ParenType(ParenType),
    TupleType(TupleType),
    NeverType(NeverType),
    PathType(PathType),
    PointerType(PointerType),
    ArrayType(ArrayType),
    SliceType(SliceType),
    ReferenceType(ReferenceType),
    PlaceholderType(PlaceholderType),
    FnPointerType(FnPointerType),
    ForType(ForType),
    ImplTraitType(ImplTraitType),
    DynTraitType(DynTraitType),
}
/// Any kind of top-level item that may appear in a module
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ModuleItem {
    StructDef(StructDef),
    UnionDef(UnionDef),
    EnumDef(EnumDef),
    FnDef(FnDef),
    TraitDef(TraitDef),
    TypeAliasDef(TypeAliasDef),
    ImplDef(ImplDef),
    UseItem(UseItem),
    ExternCrateItem(ExternCrateItem),
    ConstDef(ConstDef),
    StaticDef(StaticDef),
    Module(Module),
    MacroCall(MacroCall),
    ExternBlock(ExternBlock),
}
impl ast::NameOwner for ModuleItem {}
impl ast::AttrsOwner for ModuleItem {}
impl ast::VisibilityOwner for ModuleItem {}
/// Any kind of item that may appear in an impl block
///
/// // FIXME: impl blocks can also contain MacroCall
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AssocItem {
    FnDef(FnDef),
    TypeAliasDef(TypeAliasDef),
    ConstDef(ConstDef),
}
impl ast::NameOwner for AssocItem {}
impl ast::AttrsOwner for AssocItem {}
/// Any kind of item that may appear in an extern block
///
/// // FIXME: extern blocks can also contain MacroCall
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ExternItem {
    FnDef(FnDef),
    StaticDef(StaticDef),
}
impl ast::NameOwner for ExternItem {}
impl ast::AttrsOwner for ExternItem {}
impl ast::VisibilityOwner for ExternItem {}
/// Any kind of expression
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Expr {
    TupleExpr(TupleExpr),
    ArrayExpr(ArrayExpr),
    ParenExpr(ParenExpr),
    PathExpr(PathExpr),
    LambdaExpr(LambdaExpr),
    IfExpr(IfExpr),
    LoopExpr(LoopExpr),
    ForExpr(ForExpr),
    WhileExpr(WhileExpr),
    ContinueExpr(ContinueExpr),
    BreakExpr(BreakExpr),
    Label(Label),
    BlockExpr(BlockExpr),
    ReturnExpr(ReturnExpr),
    MatchExpr(MatchExpr),
    RecordLit(RecordLit),
    CallExpr(CallExpr),
    IndexExpr(IndexExpr),
    MethodCallExpr(MethodCallExpr),
    FieldExpr(FieldExpr),
    AwaitExpr(AwaitExpr),
    TryExpr(TryExpr),
    EffectExpr(EffectExpr),
    CastExpr(CastExpr),
    RefExpr(RefExpr),
    PrefixExpr(PrefixExpr),
    RangeExpr(RangeExpr),
    BinExpr(BinExpr),
    Literal(Literal),
    MacroCall(MacroCall),
    BoxExpr(BoxExpr),
}
impl ast::AttrsOwner for Expr {}
/// Any kind of pattern
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Pat {
    OrPat(OrPat),
    ParenPat(ParenPat),
    RefPat(RefPat),
    BoxPat(BoxPat),
    BindPat(BindPat),
    PlaceholderPat(PlaceholderPat),
    DotDotPat(DotDotPat),
    PathPat(PathPat),
    RecordPat(RecordPat),
    TupleStructPat(TupleStructPat),
    TuplePat(TuplePat),
    SlicePat(SlicePat),
    RangePat(RangePat),
    LiteralPat(LiteralPat),
    MacroPat(MacroPat),
}
/// Any kind of pattern that appears directly inside of the curly
/// braces of a record pattern
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RecordInnerPat {
    RecordFieldPat(RecordFieldPat),
    BindPat(BindPat),
}
/// Any kind of input to an attribute
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AttrInput {
    Literal(Literal),
    TokenTree(TokenTree),
}
/// Any kind of statement
/// Note: there are no empty statements, these are just represented as
/// bare semicolons without a dedicated statement ast node.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Stmt {
    LetStmt(LetStmt),
    ExprStmt(ExprStmt),
}
/// Any kind of fields list (record or tuple field lists)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FieldDefList {
    RecordFieldDefList(RecordFieldDefList),
    TupleFieldDefList(TupleFieldDefList),
}
impl AstNode for SourceFile {
    fn can_cast(kind: SyntaxKind) -> bool { kind == SOURCE_FILE }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for FnDef {
    fn can_cast(kind: SyntaxKind) -> bool { kind == FN_DEF }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for RetType {
    fn can_cast(kind: SyntaxKind) -> bool { kind == RET_TYPE }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for StructDef {
    fn can_cast(kind: SyntaxKind) -> bool { kind == STRUCT_DEF }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for UnionDef {
    fn can_cast(kind: SyntaxKind) -> bool { kind == UNION_DEF }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for RecordFieldDefList {
    fn can_cast(kind: SyntaxKind) -> bool { kind == RECORD_FIELD_DEF_LIST }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for RecordFieldDef {
    fn can_cast(kind: SyntaxKind) -> bool { kind == RECORD_FIELD_DEF }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for TupleFieldDefList {
    fn can_cast(kind: SyntaxKind) -> bool { kind == TUPLE_FIELD_DEF_LIST }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for TupleFieldDef {
    fn can_cast(kind: SyntaxKind) -> bool { kind == TUPLE_FIELD_DEF }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for EnumDef {
    fn can_cast(kind: SyntaxKind) -> bool { kind == ENUM_DEF }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for EnumVariantList {
    fn can_cast(kind: SyntaxKind) -> bool { kind == ENUM_VARIANT_LIST }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for EnumVariant {
    fn can_cast(kind: SyntaxKind) -> bool { kind == ENUM_VARIANT }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for TraitDef {
    fn can_cast(kind: SyntaxKind) -> bool { kind == TRAIT_DEF }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for Module {
    fn can_cast(kind: SyntaxKind) -> bool { kind == MODULE }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for ItemList {
    fn can_cast(kind: SyntaxKind) -> bool { kind == ITEM_LIST }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for ConstDef {
    fn can_cast(kind: SyntaxKind) -> bool { kind == CONST_DEF }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for StaticDef {
    fn can_cast(kind: SyntaxKind) -> bool { kind == STATIC_DEF }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for TypeAliasDef {
    fn can_cast(kind: SyntaxKind) -> bool { kind == TYPE_ALIAS_DEF }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for ImplDef {
    fn can_cast(kind: SyntaxKind) -> bool { kind == IMPL_DEF }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for ParenType {
    fn can_cast(kind: SyntaxKind) -> bool { kind == PAREN_TYPE }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for TupleType {
    fn can_cast(kind: SyntaxKind) -> bool { kind == TUPLE_TYPE }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for NeverType {
    fn can_cast(kind: SyntaxKind) -> bool { kind == NEVER_TYPE }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for PathType {
    fn can_cast(kind: SyntaxKind) -> bool { kind == PATH_TYPE }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for PointerType {
    fn can_cast(kind: SyntaxKind) -> bool { kind == POINTER_TYPE }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for ArrayType {
    fn can_cast(kind: SyntaxKind) -> bool { kind == ARRAY_TYPE }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for SliceType {
    fn can_cast(kind: SyntaxKind) -> bool { kind == SLICE_TYPE }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for ReferenceType {
    fn can_cast(kind: SyntaxKind) -> bool { kind == REFERENCE_TYPE }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for PlaceholderType {
    fn can_cast(kind: SyntaxKind) -> bool { kind == PLACEHOLDER_TYPE }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for FnPointerType {
    fn can_cast(kind: SyntaxKind) -> bool { kind == FN_POINTER_TYPE }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for ForType {
    fn can_cast(kind: SyntaxKind) -> bool { kind == FOR_TYPE }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for ImplTraitType {
    fn can_cast(kind: SyntaxKind) -> bool { kind == IMPL_TRAIT_TYPE }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for DynTraitType {
    fn can_cast(kind: SyntaxKind) -> bool { kind == DYN_TRAIT_TYPE }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for TupleExpr {
    fn can_cast(kind: SyntaxKind) -> bool { kind == TUPLE_EXPR }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for ArrayExpr {
    fn can_cast(kind: SyntaxKind) -> bool { kind == ARRAY_EXPR }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for ParenExpr {
    fn can_cast(kind: SyntaxKind) -> bool { kind == PAREN_EXPR }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for PathExpr {
    fn can_cast(kind: SyntaxKind) -> bool { kind == PATH_EXPR }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for LambdaExpr {
    fn can_cast(kind: SyntaxKind) -> bool { kind == LAMBDA_EXPR }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for IfExpr {
    fn can_cast(kind: SyntaxKind) -> bool { kind == IF_EXPR }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for LoopExpr {
    fn can_cast(kind: SyntaxKind) -> bool { kind == LOOP_EXPR }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for EffectExpr {
    fn can_cast(kind: SyntaxKind) -> bool { kind == EFFECT_EXPR }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for ForExpr {
    fn can_cast(kind: SyntaxKind) -> bool { kind == FOR_EXPR }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for WhileExpr {
    fn can_cast(kind: SyntaxKind) -> bool { kind == WHILE_EXPR }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for ContinueExpr {
    fn can_cast(kind: SyntaxKind) -> bool { kind == CONTINUE_EXPR }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for BreakExpr {
    fn can_cast(kind: SyntaxKind) -> bool { kind == BREAK_EXPR }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for Label {
    fn can_cast(kind: SyntaxKind) -> bool { kind == LABEL }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for BlockExpr {
    fn can_cast(kind: SyntaxKind) -> bool { kind == BLOCK_EXPR }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for ReturnExpr {
    fn can_cast(kind: SyntaxKind) -> bool { kind == RETURN_EXPR }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for CallExpr {
    fn can_cast(kind: SyntaxKind) -> bool { kind == CALL_EXPR }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for MethodCallExpr {
    fn can_cast(kind: SyntaxKind) -> bool { kind == METHOD_CALL_EXPR }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for IndexExpr {
    fn can_cast(kind: SyntaxKind) -> bool { kind == INDEX_EXPR }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for FieldExpr {
    fn can_cast(kind: SyntaxKind) -> bool { kind == FIELD_EXPR }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for AwaitExpr {
    fn can_cast(kind: SyntaxKind) -> bool { kind == AWAIT_EXPR }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for TryExpr {
    fn can_cast(kind: SyntaxKind) -> bool { kind == TRY_EXPR }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for CastExpr {
    fn can_cast(kind: SyntaxKind) -> bool { kind == CAST_EXPR }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for RefExpr {
    fn can_cast(kind: SyntaxKind) -> bool { kind == REF_EXPR }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for PrefixExpr {
    fn can_cast(kind: SyntaxKind) -> bool { kind == PREFIX_EXPR }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for BoxExpr {
    fn can_cast(kind: SyntaxKind) -> bool { kind == BOX_EXPR }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for RangeExpr {
    fn can_cast(kind: SyntaxKind) -> bool { kind == RANGE_EXPR }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for BinExpr {
    fn can_cast(kind: SyntaxKind) -> bool { kind == BIN_EXPR }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for Literal {
    fn can_cast(kind: SyntaxKind) -> bool { kind == LITERAL }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for MatchExpr {
    fn can_cast(kind: SyntaxKind) -> bool { kind == MATCH_EXPR }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for MatchArmList {
    fn can_cast(kind: SyntaxKind) -> bool { kind == MATCH_ARM_LIST }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for MatchArm {
    fn can_cast(kind: SyntaxKind) -> bool { kind == MATCH_ARM }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for MatchGuard {
    fn can_cast(kind: SyntaxKind) -> bool { kind == MATCH_GUARD }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for RecordLit {
    fn can_cast(kind: SyntaxKind) -> bool { kind == RECORD_LIT }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for RecordFieldList {
    fn can_cast(kind: SyntaxKind) -> bool { kind == RECORD_FIELD_LIST }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for RecordField {
    fn can_cast(kind: SyntaxKind) -> bool { kind == RECORD_FIELD }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for OrPat {
    fn can_cast(kind: SyntaxKind) -> bool { kind == OR_PAT }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for ParenPat {
    fn can_cast(kind: SyntaxKind) -> bool { kind == PAREN_PAT }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for RefPat {
    fn can_cast(kind: SyntaxKind) -> bool { kind == REF_PAT }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for BoxPat {
    fn can_cast(kind: SyntaxKind) -> bool { kind == BOX_PAT }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for BindPat {
    fn can_cast(kind: SyntaxKind) -> bool { kind == BIND_PAT }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for PlaceholderPat {
    fn can_cast(kind: SyntaxKind) -> bool { kind == PLACEHOLDER_PAT }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for DotDotPat {
    fn can_cast(kind: SyntaxKind) -> bool { kind == DOT_DOT_PAT }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for PathPat {
    fn can_cast(kind: SyntaxKind) -> bool { kind == PATH_PAT }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for SlicePat {
    fn can_cast(kind: SyntaxKind) -> bool { kind == SLICE_PAT }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for RangePat {
    fn can_cast(kind: SyntaxKind) -> bool { kind == RANGE_PAT }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for LiteralPat {
    fn can_cast(kind: SyntaxKind) -> bool { kind == LITERAL_PAT }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for MacroPat {
    fn can_cast(kind: SyntaxKind) -> bool { kind == MACRO_PAT }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for RecordPat {
    fn can_cast(kind: SyntaxKind) -> bool { kind == RECORD_PAT }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for RecordFieldPatList {
    fn can_cast(kind: SyntaxKind) -> bool { kind == RECORD_FIELD_PAT_LIST }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for RecordFieldPat {
    fn can_cast(kind: SyntaxKind) -> bool { kind == RECORD_FIELD_PAT }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for TupleStructPat {
    fn can_cast(kind: SyntaxKind) -> bool { kind == TUPLE_STRUCT_PAT }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for TuplePat {
    fn can_cast(kind: SyntaxKind) -> bool { kind == TUPLE_PAT }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for Visibility {
    fn can_cast(kind: SyntaxKind) -> bool { kind == VISIBILITY }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for Name {
    fn can_cast(kind: SyntaxKind) -> bool { kind == NAME }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for NameRef {
    fn can_cast(kind: SyntaxKind) -> bool { kind == NAME_REF }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for MacroCall {
    fn can_cast(kind: SyntaxKind) -> bool { kind == MACRO_CALL }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for Attr {
    fn can_cast(kind: SyntaxKind) -> bool { kind == ATTR }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for TokenTree {
    fn can_cast(kind: SyntaxKind) -> bool { kind == TOKEN_TREE }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for TypeParamList {
    fn can_cast(kind: SyntaxKind) -> bool { kind == TYPE_PARAM_LIST }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for TypeParam {
    fn can_cast(kind: SyntaxKind) -> bool { kind == TYPE_PARAM }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for ConstParam {
    fn can_cast(kind: SyntaxKind) -> bool { kind == CONST_PARAM }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for LifetimeParam {
    fn can_cast(kind: SyntaxKind) -> bool { kind == LIFETIME_PARAM }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for TypeBound {
    fn can_cast(kind: SyntaxKind) -> bool { kind == TYPE_BOUND }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for TypeBoundList {
    fn can_cast(kind: SyntaxKind) -> bool { kind == TYPE_BOUND_LIST }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for WherePred {
    fn can_cast(kind: SyntaxKind) -> bool { kind == WHERE_PRED }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for WhereClause {
    fn can_cast(kind: SyntaxKind) -> bool { kind == WHERE_CLAUSE }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for Abi {
    fn can_cast(kind: SyntaxKind) -> bool { kind == ABI }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for ExprStmt {
    fn can_cast(kind: SyntaxKind) -> bool { kind == EXPR_STMT }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for LetStmt {
    fn can_cast(kind: SyntaxKind) -> bool { kind == LET_STMT }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for Condition {
    fn can_cast(kind: SyntaxKind) -> bool { kind == CONDITION }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for ParamList {
    fn can_cast(kind: SyntaxKind) -> bool { kind == PARAM_LIST }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for SelfParam {
    fn can_cast(kind: SyntaxKind) -> bool { kind == SELF_PARAM }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for Param {
    fn can_cast(kind: SyntaxKind) -> bool { kind == PARAM }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for UseItem {
    fn can_cast(kind: SyntaxKind) -> bool { kind == USE_ITEM }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for UseTree {
    fn can_cast(kind: SyntaxKind) -> bool { kind == USE_TREE }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for Alias {
    fn can_cast(kind: SyntaxKind) -> bool { kind == ALIAS }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for UseTreeList {
    fn can_cast(kind: SyntaxKind) -> bool { kind == USE_TREE_LIST }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for ExternCrateItem {
    fn can_cast(kind: SyntaxKind) -> bool { kind == EXTERN_CRATE_ITEM }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for ArgList {
    fn can_cast(kind: SyntaxKind) -> bool { kind == ARG_LIST }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for Path {
    fn can_cast(kind: SyntaxKind) -> bool { kind == PATH }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for PathSegment {
    fn can_cast(kind: SyntaxKind) -> bool { kind == PATH_SEGMENT }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for TypeArgList {
    fn can_cast(kind: SyntaxKind) -> bool { kind == TYPE_ARG_LIST }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for TypeArg {
    fn can_cast(kind: SyntaxKind) -> bool { kind == TYPE_ARG }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for AssocTypeArg {
    fn can_cast(kind: SyntaxKind) -> bool { kind == ASSOC_TYPE_ARG }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for LifetimeArg {
    fn can_cast(kind: SyntaxKind) -> bool { kind == LIFETIME_ARG }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for ConstArg {
    fn can_cast(kind: SyntaxKind) -> bool { kind == CONST_ARG }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for MacroItems {
    fn can_cast(kind: SyntaxKind) -> bool { kind == MACRO_ITEMS }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for MacroStmts {
    fn can_cast(kind: SyntaxKind) -> bool { kind == MACRO_STMTS }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for ExternItemList {
    fn can_cast(kind: SyntaxKind) -> bool { kind == EXTERN_ITEM_LIST }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for ExternBlock {
    fn can_cast(kind: SyntaxKind) -> bool { kind == EXTERN_BLOCK }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for MetaItem {
    fn can_cast(kind: SyntaxKind) -> bool { kind == META_ITEM }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl AstNode for MacroDef {
    fn can_cast(kind: SyntaxKind) -> bool { kind == MACRO_DEF }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxNode { &self.syntax }
}
impl From<StructDef> for NominalDef {
    fn from(node: StructDef) -> NominalDef { NominalDef::StructDef(node) }
}
impl From<EnumDef> for NominalDef {
    fn from(node: EnumDef) -> NominalDef { NominalDef::EnumDef(node) }
}
impl From<UnionDef> for NominalDef {
    fn from(node: UnionDef) -> NominalDef { NominalDef::UnionDef(node) }
}
impl AstNode for NominalDef {
    fn can_cast(kind: SyntaxKind) -> bool {
        match kind {
            STRUCT_DEF | ENUM_DEF | UNION_DEF => true,
            _ => false,
        }
    }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        let res = match syntax.kind() {
            STRUCT_DEF => NominalDef::StructDef(StructDef { syntax }),
            ENUM_DEF => NominalDef::EnumDef(EnumDef { syntax }),
            UNION_DEF => NominalDef::UnionDef(UnionDef { syntax }),
            _ => return None,
        };
        Some(res)
    }
    fn syntax(&self) -> &SyntaxNode {
        match self {
            NominalDef::StructDef(it) => &it.syntax,
            NominalDef::EnumDef(it) => &it.syntax,
            NominalDef::UnionDef(it) => &it.syntax,
        }
    }
}
impl From<LifetimeParam> for GenericParam {
    fn from(node: LifetimeParam) -> GenericParam { GenericParam::LifetimeParam(node) }
}
impl From<TypeParam> for GenericParam {
    fn from(node: TypeParam) -> GenericParam { GenericParam::TypeParam(node) }
}
impl From<ConstParam> for GenericParam {
    fn from(node: ConstParam) -> GenericParam { GenericParam::ConstParam(node) }
}
impl AstNode for GenericParam {
    fn can_cast(kind: SyntaxKind) -> bool {
        match kind {
            LIFETIME_PARAM | TYPE_PARAM | CONST_PARAM => true,
            _ => false,
        }
    }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        let res = match syntax.kind() {
            LIFETIME_PARAM => GenericParam::LifetimeParam(LifetimeParam { syntax }),
            TYPE_PARAM => GenericParam::TypeParam(TypeParam { syntax }),
            CONST_PARAM => GenericParam::ConstParam(ConstParam { syntax }),
            _ => return None,
        };
        Some(res)
    }
    fn syntax(&self) -> &SyntaxNode {
        match self {
            GenericParam::LifetimeParam(it) => &it.syntax,
            GenericParam::TypeParam(it) => &it.syntax,
            GenericParam::ConstParam(it) => &it.syntax,
        }
    }
}
impl From<LifetimeArg> for GenericArg {
    fn from(node: LifetimeArg) -> GenericArg { GenericArg::LifetimeArg(node) }
}
impl From<TypeArg> for GenericArg {
    fn from(node: TypeArg) -> GenericArg { GenericArg::TypeArg(node) }
}
impl From<ConstArg> for GenericArg {
    fn from(node: ConstArg) -> GenericArg { GenericArg::ConstArg(node) }
}
impl From<AssocTypeArg> for GenericArg {
    fn from(node: AssocTypeArg) -> GenericArg { GenericArg::AssocTypeArg(node) }
}
impl AstNode for GenericArg {
    fn can_cast(kind: SyntaxKind) -> bool {
        match kind {
            LIFETIME_ARG | TYPE_ARG | CONST_ARG | ASSOC_TYPE_ARG => true,
            _ => false,
        }
    }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        let res = match syntax.kind() {
            LIFETIME_ARG => GenericArg::LifetimeArg(LifetimeArg { syntax }),
            TYPE_ARG => GenericArg::TypeArg(TypeArg { syntax }),
            CONST_ARG => GenericArg::ConstArg(ConstArg { syntax }),
            ASSOC_TYPE_ARG => GenericArg::AssocTypeArg(AssocTypeArg { syntax }),
            _ => return None,
        };
        Some(res)
    }
    fn syntax(&self) -> &SyntaxNode {
        match self {
            GenericArg::LifetimeArg(it) => &it.syntax,
            GenericArg::TypeArg(it) => &it.syntax,
            GenericArg::ConstArg(it) => &it.syntax,
            GenericArg::AssocTypeArg(it) => &it.syntax,
        }
    }
}
impl From<ParenType> for TypeRef {
    fn from(node: ParenType) -> TypeRef { TypeRef::ParenType(node) }
}
impl From<TupleType> for TypeRef {
    fn from(node: TupleType) -> TypeRef { TypeRef::TupleType(node) }
}
impl From<NeverType> for TypeRef {
    fn from(node: NeverType) -> TypeRef { TypeRef::NeverType(node) }
}
impl From<PathType> for TypeRef {
    fn from(node: PathType) -> TypeRef { TypeRef::PathType(node) }
}
impl From<PointerType> for TypeRef {
    fn from(node: PointerType) -> TypeRef { TypeRef::PointerType(node) }
}
impl From<ArrayType> for TypeRef {
    fn from(node: ArrayType) -> TypeRef { TypeRef::ArrayType(node) }
}
impl From<SliceType> for TypeRef {
    fn from(node: SliceType) -> TypeRef { TypeRef::SliceType(node) }
}
impl From<ReferenceType> for TypeRef {
    fn from(node: ReferenceType) -> TypeRef { TypeRef::ReferenceType(node) }
}
impl From<PlaceholderType> for TypeRef {
    fn from(node: PlaceholderType) -> TypeRef { TypeRef::PlaceholderType(node) }
}
impl From<FnPointerType> for TypeRef {
    fn from(node: FnPointerType) -> TypeRef { TypeRef::FnPointerType(node) }
}
impl From<ForType> for TypeRef {
    fn from(node: ForType) -> TypeRef { TypeRef::ForType(node) }
}
impl From<ImplTraitType> for TypeRef {
    fn from(node: ImplTraitType) -> TypeRef { TypeRef::ImplTraitType(node) }
}
impl From<DynTraitType> for TypeRef {
    fn from(node: DynTraitType) -> TypeRef { TypeRef::DynTraitType(node) }
}
impl AstNode for TypeRef {
    fn can_cast(kind: SyntaxKind) -> bool {
        match kind {
            PAREN_TYPE | TUPLE_TYPE | NEVER_TYPE | PATH_TYPE | POINTER_TYPE | ARRAY_TYPE
            | SLICE_TYPE | REFERENCE_TYPE | PLACEHOLDER_TYPE | FN_POINTER_TYPE | FOR_TYPE
            | IMPL_TRAIT_TYPE | DYN_TRAIT_TYPE => true,
            _ => false,
        }
    }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        let res = match syntax.kind() {
            PAREN_TYPE => TypeRef::ParenType(ParenType { syntax }),
            TUPLE_TYPE => TypeRef::TupleType(TupleType { syntax }),
            NEVER_TYPE => TypeRef::NeverType(NeverType { syntax }),
            PATH_TYPE => TypeRef::PathType(PathType { syntax }),
            POINTER_TYPE => TypeRef::PointerType(PointerType { syntax }),
            ARRAY_TYPE => TypeRef::ArrayType(ArrayType { syntax }),
            SLICE_TYPE => TypeRef::SliceType(SliceType { syntax }),
            REFERENCE_TYPE => TypeRef::ReferenceType(ReferenceType { syntax }),
            PLACEHOLDER_TYPE => TypeRef::PlaceholderType(PlaceholderType { syntax }),
            FN_POINTER_TYPE => TypeRef::FnPointerType(FnPointerType { syntax }),
            FOR_TYPE => TypeRef::ForType(ForType { syntax }),
            IMPL_TRAIT_TYPE => TypeRef::ImplTraitType(ImplTraitType { syntax }),
            DYN_TRAIT_TYPE => TypeRef::DynTraitType(DynTraitType { syntax }),
            _ => return None,
        };
        Some(res)
    }
    fn syntax(&self) -> &SyntaxNode {
        match self {
            TypeRef::ParenType(it) => &it.syntax,
            TypeRef::TupleType(it) => &it.syntax,
            TypeRef::NeverType(it) => &it.syntax,
            TypeRef::PathType(it) => &it.syntax,
            TypeRef::PointerType(it) => &it.syntax,
            TypeRef::ArrayType(it) => &it.syntax,
            TypeRef::SliceType(it) => &it.syntax,
            TypeRef::ReferenceType(it) => &it.syntax,
            TypeRef::PlaceholderType(it) => &it.syntax,
            TypeRef::FnPointerType(it) => &it.syntax,
            TypeRef::ForType(it) => &it.syntax,
            TypeRef::ImplTraitType(it) => &it.syntax,
            TypeRef::DynTraitType(it) => &it.syntax,
        }
    }
}
impl From<StructDef> for ModuleItem {
    fn from(node: StructDef) -> ModuleItem { ModuleItem::StructDef(node) }
}
impl From<UnionDef> for ModuleItem {
    fn from(node: UnionDef) -> ModuleItem { ModuleItem::UnionDef(node) }
}
impl From<EnumDef> for ModuleItem {
    fn from(node: EnumDef) -> ModuleItem { ModuleItem::EnumDef(node) }
}
impl From<FnDef> for ModuleItem {
    fn from(node: FnDef) -> ModuleItem { ModuleItem::FnDef(node) }
}
impl From<TraitDef> for ModuleItem {
    fn from(node: TraitDef) -> ModuleItem { ModuleItem::TraitDef(node) }
}
impl From<TypeAliasDef> for ModuleItem {
    fn from(node: TypeAliasDef) -> ModuleItem { ModuleItem::TypeAliasDef(node) }
}
impl From<ImplDef> for ModuleItem {
    fn from(node: ImplDef) -> ModuleItem { ModuleItem::ImplDef(node) }
}
impl From<UseItem> for ModuleItem {
    fn from(node: UseItem) -> ModuleItem { ModuleItem::UseItem(node) }
}
impl From<ExternCrateItem> for ModuleItem {
    fn from(node: ExternCrateItem) -> ModuleItem { ModuleItem::ExternCrateItem(node) }
}
impl From<ConstDef> for ModuleItem {
    fn from(node: ConstDef) -> ModuleItem { ModuleItem::ConstDef(node) }
}
impl From<StaticDef> for ModuleItem {
    fn from(node: StaticDef) -> ModuleItem { ModuleItem::StaticDef(node) }
}
impl From<Module> for ModuleItem {
    fn from(node: Module) -> ModuleItem { ModuleItem::Module(node) }
}
impl From<MacroCall> for ModuleItem {
    fn from(node: MacroCall) -> ModuleItem { ModuleItem::MacroCall(node) }
}
impl From<ExternBlock> for ModuleItem {
    fn from(node: ExternBlock) -> ModuleItem { ModuleItem::ExternBlock(node) }
}
impl AstNode for ModuleItem {
    fn can_cast(kind: SyntaxKind) -> bool {
        match kind {
            STRUCT_DEF | UNION_DEF | ENUM_DEF | FN_DEF | TRAIT_DEF | TYPE_ALIAS_DEF | IMPL_DEF
            | USE_ITEM | EXTERN_CRATE_ITEM | CONST_DEF | STATIC_DEF | MODULE | MACRO_CALL
            | EXTERN_BLOCK => true,
            _ => false,
        }
    }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        let res = match syntax.kind() {
            STRUCT_DEF => ModuleItem::StructDef(StructDef { syntax }),
            UNION_DEF => ModuleItem::UnionDef(UnionDef { syntax }),
            ENUM_DEF => ModuleItem::EnumDef(EnumDef { syntax }),
            FN_DEF => ModuleItem::FnDef(FnDef { syntax }),
            TRAIT_DEF => ModuleItem::TraitDef(TraitDef { syntax }),
            TYPE_ALIAS_DEF => ModuleItem::TypeAliasDef(TypeAliasDef { syntax }),
            IMPL_DEF => ModuleItem::ImplDef(ImplDef { syntax }),
            USE_ITEM => ModuleItem::UseItem(UseItem { syntax }),
            EXTERN_CRATE_ITEM => ModuleItem::ExternCrateItem(ExternCrateItem { syntax }),
            CONST_DEF => ModuleItem::ConstDef(ConstDef { syntax }),
            STATIC_DEF => ModuleItem::StaticDef(StaticDef { syntax }),
            MODULE => ModuleItem::Module(Module { syntax }),
            MACRO_CALL => ModuleItem::MacroCall(MacroCall { syntax }),
            EXTERN_BLOCK => ModuleItem::ExternBlock(ExternBlock { syntax }),
            _ => return None,
        };
        Some(res)
    }
    fn syntax(&self) -> &SyntaxNode {
        match self {
            ModuleItem::StructDef(it) => &it.syntax,
            ModuleItem::UnionDef(it) => &it.syntax,
            ModuleItem::EnumDef(it) => &it.syntax,
            ModuleItem::FnDef(it) => &it.syntax,
            ModuleItem::TraitDef(it) => &it.syntax,
            ModuleItem::TypeAliasDef(it) => &it.syntax,
            ModuleItem::ImplDef(it) => &it.syntax,
            ModuleItem::UseItem(it) => &it.syntax,
            ModuleItem::ExternCrateItem(it) => &it.syntax,
            ModuleItem::ConstDef(it) => &it.syntax,
            ModuleItem::StaticDef(it) => &it.syntax,
            ModuleItem::Module(it) => &it.syntax,
            ModuleItem::MacroCall(it) => &it.syntax,
            ModuleItem::ExternBlock(it) => &it.syntax,
        }
    }
}
impl From<FnDef> for AssocItem {
    fn from(node: FnDef) -> AssocItem { AssocItem::FnDef(node) }
}
impl From<TypeAliasDef> for AssocItem {
    fn from(node: TypeAliasDef) -> AssocItem { AssocItem::TypeAliasDef(node) }
}
impl From<ConstDef> for AssocItem {
    fn from(node: ConstDef) -> AssocItem { AssocItem::ConstDef(node) }
}
impl AstNode for AssocItem {
    fn can_cast(kind: SyntaxKind) -> bool {
        match kind {
            FN_DEF | TYPE_ALIAS_DEF | CONST_DEF => true,
            _ => false,
        }
    }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        let res = match syntax.kind() {
            FN_DEF => AssocItem::FnDef(FnDef { syntax }),
            TYPE_ALIAS_DEF => AssocItem::TypeAliasDef(TypeAliasDef { syntax }),
            CONST_DEF => AssocItem::ConstDef(ConstDef { syntax }),
            _ => return None,
        };
        Some(res)
    }
    fn syntax(&self) -> &SyntaxNode {
        match self {
            AssocItem::FnDef(it) => &it.syntax,
            AssocItem::TypeAliasDef(it) => &it.syntax,
            AssocItem::ConstDef(it) => &it.syntax,
        }
    }
}
impl From<FnDef> for ExternItem {
    fn from(node: FnDef) -> ExternItem { ExternItem::FnDef(node) }
}
impl From<StaticDef> for ExternItem {
    fn from(node: StaticDef) -> ExternItem { ExternItem::StaticDef(node) }
}
impl AstNode for ExternItem {
    fn can_cast(kind: SyntaxKind) -> bool {
        match kind {
            FN_DEF | STATIC_DEF => true,
            _ => false,
        }
    }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        let res = match syntax.kind() {
            FN_DEF => ExternItem::FnDef(FnDef { syntax }),
            STATIC_DEF => ExternItem::StaticDef(StaticDef { syntax }),
            _ => return None,
        };
        Some(res)
    }
    fn syntax(&self) -> &SyntaxNode {
        match self {
            ExternItem::FnDef(it) => &it.syntax,
            ExternItem::StaticDef(it) => &it.syntax,
        }
    }
}
impl From<TupleExpr> for Expr {
    fn from(node: TupleExpr) -> Expr { Expr::TupleExpr(node) }
}
impl From<ArrayExpr> for Expr {
    fn from(node: ArrayExpr) -> Expr { Expr::ArrayExpr(node) }
}
impl From<ParenExpr> for Expr {
    fn from(node: ParenExpr) -> Expr { Expr::ParenExpr(node) }
}
impl From<PathExpr> for Expr {
    fn from(node: PathExpr) -> Expr { Expr::PathExpr(node) }
}
impl From<LambdaExpr> for Expr {
    fn from(node: LambdaExpr) -> Expr { Expr::LambdaExpr(node) }
}
impl From<IfExpr> for Expr {
    fn from(node: IfExpr) -> Expr { Expr::IfExpr(node) }
}
impl From<LoopExpr> for Expr {
    fn from(node: LoopExpr) -> Expr { Expr::LoopExpr(node) }
}
impl From<ForExpr> for Expr {
    fn from(node: ForExpr) -> Expr { Expr::ForExpr(node) }
}
impl From<WhileExpr> for Expr {
    fn from(node: WhileExpr) -> Expr { Expr::WhileExpr(node) }
}
impl From<ContinueExpr> for Expr {
    fn from(node: ContinueExpr) -> Expr { Expr::ContinueExpr(node) }
}
impl From<BreakExpr> for Expr {
    fn from(node: BreakExpr) -> Expr { Expr::BreakExpr(node) }
}
impl From<Label> for Expr {
    fn from(node: Label) -> Expr { Expr::Label(node) }
}
impl From<BlockExpr> for Expr {
    fn from(node: BlockExpr) -> Expr { Expr::BlockExpr(node) }
}
impl From<ReturnExpr> for Expr {
    fn from(node: ReturnExpr) -> Expr { Expr::ReturnExpr(node) }
}
impl From<MatchExpr> for Expr {
    fn from(node: MatchExpr) -> Expr { Expr::MatchExpr(node) }
}
impl From<RecordLit> for Expr {
    fn from(node: RecordLit) -> Expr { Expr::RecordLit(node) }
}
impl From<CallExpr> for Expr {
    fn from(node: CallExpr) -> Expr { Expr::CallExpr(node) }
}
impl From<IndexExpr> for Expr {
    fn from(node: IndexExpr) -> Expr { Expr::IndexExpr(node) }
}
impl From<MethodCallExpr> for Expr {
    fn from(node: MethodCallExpr) -> Expr { Expr::MethodCallExpr(node) }
}
impl From<FieldExpr> for Expr {
    fn from(node: FieldExpr) -> Expr { Expr::FieldExpr(node) }
}
impl From<AwaitExpr> for Expr {
    fn from(node: AwaitExpr) -> Expr { Expr::AwaitExpr(node) }
}
impl From<TryExpr> for Expr {
    fn from(node: TryExpr) -> Expr { Expr::TryExpr(node) }
}
impl From<EffectExpr> for Expr {
    fn from(node: EffectExpr) -> Expr { Expr::EffectExpr(node) }
}
impl From<CastExpr> for Expr {
    fn from(node: CastExpr) -> Expr { Expr::CastExpr(node) }
}
impl From<RefExpr> for Expr {
    fn from(node: RefExpr) -> Expr { Expr::RefExpr(node) }
}
impl From<PrefixExpr> for Expr {
    fn from(node: PrefixExpr) -> Expr { Expr::PrefixExpr(node) }
}
impl From<RangeExpr> for Expr {
    fn from(node: RangeExpr) -> Expr { Expr::RangeExpr(node) }
}
impl From<BinExpr> for Expr {
    fn from(node: BinExpr) -> Expr { Expr::BinExpr(node) }
}
impl From<Literal> for Expr {
    fn from(node: Literal) -> Expr { Expr::Literal(node) }
}
impl From<MacroCall> for Expr {
    fn from(node: MacroCall) -> Expr { Expr::MacroCall(node) }
}
impl From<BoxExpr> for Expr {
    fn from(node: BoxExpr) -> Expr { Expr::BoxExpr(node) }
}
impl AstNode for Expr {
    fn can_cast(kind: SyntaxKind) -> bool {
        match kind {
            TUPLE_EXPR | ARRAY_EXPR | PAREN_EXPR | PATH_EXPR | LAMBDA_EXPR | IF_EXPR
            | LOOP_EXPR | FOR_EXPR | WHILE_EXPR | CONTINUE_EXPR | BREAK_EXPR | LABEL
            | BLOCK_EXPR | RETURN_EXPR | MATCH_EXPR | RECORD_LIT | CALL_EXPR | INDEX_EXPR
            | METHOD_CALL_EXPR | FIELD_EXPR | AWAIT_EXPR | TRY_EXPR | EFFECT_EXPR | CAST_EXPR
            | REF_EXPR | PREFIX_EXPR | RANGE_EXPR | BIN_EXPR | LITERAL | MACRO_CALL | BOX_EXPR => {
                true
            }
            _ => false,
        }
    }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        let res = match syntax.kind() {
            TUPLE_EXPR => Expr::TupleExpr(TupleExpr { syntax }),
            ARRAY_EXPR => Expr::ArrayExpr(ArrayExpr { syntax }),
            PAREN_EXPR => Expr::ParenExpr(ParenExpr { syntax }),
            PATH_EXPR => Expr::PathExpr(PathExpr { syntax }),
            LAMBDA_EXPR => Expr::LambdaExpr(LambdaExpr { syntax }),
            IF_EXPR => Expr::IfExpr(IfExpr { syntax }),
            LOOP_EXPR => Expr::LoopExpr(LoopExpr { syntax }),
            FOR_EXPR => Expr::ForExpr(ForExpr { syntax }),
            WHILE_EXPR => Expr::WhileExpr(WhileExpr { syntax }),
            CONTINUE_EXPR => Expr::ContinueExpr(ContinueExpr { syntax }),
            BREAK_EXPR => Expr::BreakExpr(BreakExpr { syntax }),
            LABEL => Expr::Label(Label { syntax }),
            BLOCK_EXPR => Expr::BlockExpr(BlockExpr { syntax }),
            RETURN_EXPR => Expr::ReturnExpr(ReturnExpr { syntax }),
            MATCH_EXPR => Expr::MatchExpr(MatchExpr { syntax }),
            RECORD_LIT => Expr::RecordLit(RecordLit { syntax }),
            CALL_EXPR => Expr::CallExpr(CallExpr { syntax }),
            INDEX_EXPR => Expr::IndexExpr(IndexExpr { syntax }),
            METHOD_CALL_EXPR => Expr::MethodCallExpr(MethodCallExpr { syntax }),
            FIELD_EXPR => Expr::FieldExpr(FieldExpr { syntax }),
            AWAIT_EXPR => Expr::AwaitExpr(AwaitExpr { syntax }),
            TRY_EXPR => Expr::TryExpr(TryExpr { syntax }),
            EFFECT_EXPR => Expr::EffectExpr(EffectExpr { syntax }),
            CAST_EXPR => Expr::CastExpr(CastExpr { syntax }),
            REF_EXPR => Expr::RefExpr(RefExpr { syntax }),
            PREFIX_EXPR => Expr::PrefixExpr(PrefixExpr { syntax }),
            RANGE_EXPR => Expr::RangeExpr(RangeExpr { syntax }),
            BIN_EXPR => Expr::BinExpr(BinExpr { syntax }),
            LITERAL => Expr::Literal(Literal { syntax }),
            MACRO_CALL => Expr::MacroCall(MacroCall { syntax }),
            BOX_EXPR => Expr::BoxExpr(BoxExpr { syntax }),
            _ => return None,
        };
        Some(res)
    }
    fn syntax(&self) -> &SyntaxNode {
        match self {
            Expr::TupleExpr(it) => &it.syntax,
            Expr::ArrayExpr(it) => &it.syntax,
            Expr::ParenExpr(it) => &it.syntax,
            Expr::PathExpr(it) => &it.syntax,
            Expr::LambdaExpr(it) => &it.syntax,
            Expr::IfExpr(it) => &it.syntax,
            Expr::LoopExpr(it) => &it.syntax,
            Expr::ForExpr(it) => &it.syntax,
            Expr::WhileExpr(it) => &it.syntax,
            Expr::ContinueExpr(it) => &it.syntax,
            Expr::BreakExpr(it) => &it.syntax,
            Expr::Label(it) => &it.syntax,
            Expr::BlockExpr(it) => &it.syntax,
            Expr::ReturnExpr(it) => &it.syntax,
            Expr::MatchExpr(it) => &it.syntax,
            Expr::RecordLit(it) => &it.syntax,
            Expr::CallExpr(it) => &it.syntax,
            Expr::IndexExpr(it) => &it.syntax,
            Expr::MethodCallExpr(it) => &it.syntax,
            Expr::FieldExpr(it) => &it.syntax,
            Expr::AwaitExpr(it) => &it.syntax,
            Expr::TryExpr(it) => &it.syntax,
            Expr::EffectExpr(it) => &it.syntax,
            Expr::CastExpr(it) => &it.syntax,
            Expr::RefExpr(it) => &it.syntax,
            Expr::PrefixExpr(it) => &it.syntax,
            Expr::RangeExpr(it) => &it.syntax,
            Expr::BinExpr(it) => &it.syntax,
            Expr::Literal(it) => &it.syntax,
            Expr::MacroCall(it) => &it.syntax,
            Expr::BoxExpr(it) => &it.syntax,
        }
    }
}
impl From<OrPat> for Pat {
    fn from(node: OrPat) -> Pat { Pat::OrPat(node) }
}
impl From<ParenPat> for Pat {
    fn from(node: ParenPat) -> Pat { Pat::ParenPat(node) }
}
impl From<RefPat> for Pat {
    fn from(node: RefPat) -> Pat { Pat::RefPat(node) }
}
impl From<BoxPat> for Pat {
    fn from(node: BoxPat) -> Pat { Pat::BoxPat(node) }
}
impl From<BindPat> for Pat {
    fn from(node: BindPat) -> Pat { Pat::BindPat(node) }
}
impl From<PlaceholderPat> for Pat {
    fn from(node: PlaceholderPat) -> Pat { Pat::PlaceholderPat(node) }
}
impl From<DotDotPat> for Pat {
    fn from(node: DotDotPat) -> Pat { Pat::DotDotPat(node) }
}
impl From<PathPat> for Pat {
    fn from(node: PathPat) -> Pat { Pat::PathPat(node) }
}
impl From<RecordPat> for Pat {
    fn from(node: RecordPat) -> Pat { Pat::RecordPat(node) }
}
impl From<TupleStructPat> for Pat {
    fn from(node: TupleStructPat) -> Pat { Pat::TupleStructPat(node) }
}
impl From<TuplePat> for Pat {
    fn from(node: TuplePat) -> Pat { Pat::TuplePat(node) }
}
impl From<SlicePat> for Pat {
    fn from(node: SlicePat) -> Pat { Pat::SlicePat(node) }
}
impl From<RangePat> for Pat {
    fn from(node: RangePat) -> Pat { Pat::RangePat(node) }
}
impl From<LiteralPat> for Pat {
    fn from(node: LiteralPat) -> Pat { Pat::LiteralPat(node) }
}
impl From<MacroPat> for Pat {
    fn from(node: MacroPat) -> Pat { Pat::MacroPat(node) }
}
impl AstNode for Pat {
    fn can_cast(kind: SyntaxKind) -> bool {
        match kind {
            OR_PAT | PAREN_PAT | REF_PAT | BOX_PAT | BIND_PAT | PLACEHOLDER_PAT | DOT_DOT_PAT
            | PATH_PAT | RECORD_PAT | TUPLE_STRUCT_PAT | TUPLE_PAT | SLICE_PAT | RANGE_PAT
            | LITERAL_PAT | MACRO_PAT => true,
            _ => false,
        }
    }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        let res = match syntax.kind() {
            OR_PAT => Pat::OrPat(OrPat { syntax }),
            PAREN_PAT => Pat::ParenPat(ParenPat { syntax }),
            REF_PAT => Pat::RefPat(RefPat { syntax }),
            BOX_PAT => Pat::BoxPat(BoxPat { syntax }),
            BIND_PAT => Pat::BindPat(BindPat { syntax }),
            PLACEHOLDER_PAT => Pat::PlaceholderPat(PlaceholderPat { syntax }),
            DOT_DOT_PAT => Pat::DotDotPat(DotDotPat { syntax }),
            PATH_PAT => Pat::PathPat(PathPat { syntax }),
            RECORD_PAT => Pat::RecordPat(RecordPat { syntax }),
            TUPLE_STRUCT_PAT => Pat::TupleStructPat(TupleStructPat { syntax }),
            TUPLE_PAT => Pat::TuplePat(TuplePat { syntax }),
            SLICE_PAT => Pat::SlicePat(SlicePat { syntax }),
            RANGE_PAT => Pat::RangePat(RangePat { syntax }),
            LITERAL_PAT => Pat::LiteralPat(LiteralPat { syntax }),
            MACRO_PAT => Pat::MacroPat(MacroPat { syntax }),
            _ => return None,
        };
        Some(res)
    }
    fn syntax(&self) -> &SyntaxNode {
        match self {
            Pat::OrPat(it) => &it.syntax,
            Pat::ParenPat(it) => &it.syntax,
            Pat::RefPat(it) => &it.syntax,
            Pat::BoxPat(it) => &it.syntax,
            Pat::BindPat(it) => &it.syntax,
            Pat::PlaceholderPat(it) => &it.syntax,
            Pat::DotDotPat(it) => &it.syntax,
            Pat::PathPat(it) => &it.syntax,
            Pat::RecordPat(it) => &it.syntax,
            Pat::TupleStructPat(it) => &it.syntax,
            Pat::TuplePat(it) => &it.syntax,
            Pat::SlicePat(it) => &it.syntax,
            Pat::RangePat(it) => &it.syntax,
            Pat::LiteralPat(it) => &it.syntax,
            Pat::MacroPat(it) => &it.syntax,
        }
    }
}
impl From<RecordFieldPat> for RecordInnerPat {
    fn from(node: RecordFieldPat) -> RecordInnerPat { RecordInnerPat::RecordFieldPat(node) }
}
impl From<BindPat> for RecordInnerPat {
    fn from(node: BindPat) -> RecordInnerPat { RecordInnerPat::BindPat(node) }
}
impl AstNode for RecordInnerPat {
    fn can_cast(kind: SyntaxKind) -> bool {
        match kind {
            RECORD_FIELD_PAT | BIND_PAT => true,
            _ => false,
        }
    }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        let res = match syntax.kind() {
            RECORD_FIELD_PAT => RecordInnerPat::RecordFieldPat(RecordFieldPat { syntax }),
            BIND_PAT => RecordInnerPat::BindPat(BindPat { syntax }),
            _ => return None,
        };
        Some(res)
    }
    fn syntax(&self) -> &SyntaxNode {
        match self {
            RecordInnerPat::RecordFieldPat(it) => &it.syntax,
            RecordInnerPat::BindPat(it) => &it.syntax,
        }
    }
}
impl From<Literal> for AttrInput {
    fn from(node: Literal) -> AttrInput { AttrInput::Literal(node) }
}
impl From<TokenTree> for AttrInput {
    fn from(node: TokenTree) -> AttrInput { AttrInput::TokenTree(node) }
}
impl AstNode for AttrInput {
    fn can_cast(kind: SyntaxKind) -> bool {
        match kind {
            LITERAL | TOKEN_TREE => true,
            _ => false,
        }
    }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        let res = match syntax.kind() {
            LITERAL => AttrInput::Literal(Literal { syntax }),
            TOKEN_TREE => AttrInput::TokenTree(TokenTree { syntax }),
            _ => return None,
        };
        Some(res)
    }
    fn syntax(&self) -> &SyntaxNode {
        match self {
            AttrInput::Literal(it) => &it.syntax,
            AttrInput::TokenTree(it) => &it.syntax,
        }
    }
}
impl From<LetStmt> for Stmt {
    fn from(node: LetStmt) -> Stmt { Stmt::LetStmt(node) }
}
impl From<ExprStmt> for Stmt {
    fn from(node: ExprStmt) -> Stmt { Stmt::ExprStmt(node) }
}
impl AstNode for Stmt {
    fn can_cast(kind: SyntaxKind) -> bool {
        match kind {
            LET_STMT | EXPR_STMT => true,
            _ => false,
        }
    }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        let res = match syntax.kind() {
            LET_STMT => Stmt::LetStmt(LetStmt { syntax }),
            EXPR_STMT => Stmt::ExprStmt(ExprStmt { syntax }),
            _ => return None,
        };
        Some(res)
    }
    fn syntax(&self) -> &SyntaxNode {
        match self {
            Stmt::LetStmt(it) => &it.syntax,
            Stmt::ExprStmt(it) => &it.syntax,
        }
    }
}
impl From<RecordFieldDefList> for FieldDefList {
    fn from(node: RecordFieldDefList) -> FieldDefList { FieldDefList::RecordFieldDefList(node) }
}
impl From<TupleFieldDefList> for FieldDefList {
    fn from(node: TupleFieldDefList) -> FieldDefList { FieldDefList::TupleFieldDefList(node) }
}
impl AstNode for FieldDefList {
    fn can_cast(kind: SyntaxKind) -> bool {
        match kind {
            RECORD_FIELD_DEF_LIST | TUPLE_FIELD_DEF_LIST => true,
            _ => false,
        }
    }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        let res = match syntax.kind() {
            RECORD_FIELD_DEF_LIST => {
                FieldDefList::RecordFieldDefList(RecordFieldDefList { syntax })
            }
            TUPLE_FIELD_DEF_LIST => FieldDefList::TupleFieldDefList(TupleFieldDefList { syntax }),
            _ => return None,
        };
        Some(res)
    }
    fn syntax(&self) -> &SyntaxNode {
        match self {
            FieldDefList::RecordFieldDefList(it) => &it.syntax,
            FieldDefList::TupleFieldDefList(it) => &it.syntax,
        }
    }
}
impl std::fmt::Display for NominalDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for GenericParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for GenericArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for TypeRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for ModuleItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for AssocItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for ExternItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for Pat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for RecordInnerPat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for AttrInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for Stmt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for FieldDefList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for SourceFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for FnDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for RetType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for StructDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for UnionDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for RecordFieldDefList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for RecordFieldDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for TupleFieldDefList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for TupleFieldDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for EnumDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for EnumVariantList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for EnumVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for TraitDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for Module {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for ItemList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for ConstDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for StaticDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for TypeAliasDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for ImplDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for ParenType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for TupleType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for NeverType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for PathType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for PointerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for ArrayType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for SliceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for ReferenceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for PlaceholderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for FnPointerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for ForType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for ImplTraitType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for DynTraitType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for TupleExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for ArrayExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for ParenExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for PathExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for LambdaExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for IfExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for LoopExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for EffectExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for ForExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for WhileExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for ContinueExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for BreakExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for Label {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for BlockExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for ReturnExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for CallExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for MethodCallExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for IndexExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for FieldExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for AwaitExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for TryExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for CastExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for RefExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for PrefixExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for BoxExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for RangeExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for BinExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for Literal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for MatchExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for MatchArmList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for MatchArm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for MatchGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for RecordLit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for RecordFieldList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for RecordField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for OrPat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for ParenPat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for RefPat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for BoxPat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for BindPat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for PlaceholderPat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for DotDotPat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for PathPat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for SlicePat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for RangePat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for LiteralPat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for MacroPat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for RecordPat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for RecordFieldPatList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for RecordFieldPat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for TupleStructPat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for TuplePat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for Visibility {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for Name {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for NameRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for MacroCall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for Attr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for TokenTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for TypeParamList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for TypeParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for ConstParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for LifetimeParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for TypeBound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for TypeBoundList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for WherePred {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for WhereClause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for Abi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for ExprStmt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for LetStmt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for Condition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for ParamList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for SelfParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for Param {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for UseItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for UseTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for Alias {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for UseTreeList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for ExternCrateItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for ArgList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for Path {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for PathSegment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for TypeArgList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for TypeArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for AssocTypeArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for LifetimeArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for ConstArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for MacroItems {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for MacroStmts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for ExternItemList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for ExternBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for MetaItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl std::fmt::Display for MacroDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
