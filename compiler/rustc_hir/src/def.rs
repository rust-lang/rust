use std::array::IntoIter;
use std::fmt::Debug;

use rustc_ast as ast;
use rustc_ast::NodeId;
use rustc_data_structures::stable_hasher::ToStableHashKey;
use rustc_data_structures::unord::UnordMap;
use rustc_macros::{Decodable, Encodable, HashStable_Generic};
use rustc_span::Symbol;
use rustc_span::def_id::{DefId, LocalDefId};
use rustc_span::hygiene::MacroKind;

use crate::definitions::DefPathData;
use crate::hir;

/// Encodes if a `DefKind::Ctor` is the constructor of an enum variant or a struct.
#[derive(Clone, Copy, PartialEq, Eq, Encodable, Decodable, Hash, Debug, HashStable_Generic)]
pub enum CtorOf {
    /// This `DefKind::Ctor` is a synthesized constructor of a tuple or unit struct.
    Struct,
    /// This `DefKind::Ctor` is a synthesized constructor of a tuple or unit variant.
    Variant,
}

/// What kind of constructor something is.
#[derive(Clone, Copy, PartialEq, Eq, Encodable, Decodable, Hash, Debug, HashStable_Generic)]
pub enum CtorKind {
    /// Constructor function automatically created by a tuple struct/variant.
    Fn,
    /// Constructor constant automatically created by a unit struct/variant.
    Const,
}

/// An attribute that is not a macro; e.g., `#[inline]` or `#[rustfmt::skip]`.
#[derive(Clone, Copy, PartialEq, Eq, Encodable, Decodable, Hash, Debug, HashStable_Generic)]
pub enum NonMacroAttrKind {
    /// Single-segment attribute defined by the language (`#[inline]`)
    Builtin(Symbol),
    /// Multi-segment custom attribute living in a "tool module" (`#[rustfmt::skip]`).
    Tool,
    /// Single-segment custom attribute registered by a derive macro (`#[serde(default)]`).
    DeriveHelper,
    /// Single-segment custom attribute registered by a derive macro
    /// but used before that derive macro was expanded (deprecated).
    DeriveHelperCompat,
}

/// What kind of definition something is; e.g., `mod` vs `struct`.
/// `enum DefPathData` may need to be updated if a new variant is added here.
#[derive(Clone, Copy, PartialEq, Eq, Encodable, Decodable, Hash, Debug, HashStable_Generic)]
pub enum DefKind {
    // Type namespace
    Mod,
    /// Refers to the struct itself, [`DefKind::Ctor`] refers to its constructor if it exists.
    Struct,
    Union,
    Enum,
    /// Refers to the variant itself, [`DefKind::Ctor`] refers to its constructor if it exists.
    Variant,
    Trait,
    /// Type alias: `type Foo = Bar;`
    TyAlias,
    /// Type from an `extern` block.
    ForeignTy,
    /// Trait alias: `trait IntIterator = Iterator<Item = i32>;`
    TraitAlias,
    /// Associated type: `trait MyTrait { type Assoc; }`
    AssocTy,
    /// Type parameter: the `T` in `struct Vec<T> { ... }`
    TyParam,

    // Value namespace
    Fn,
    Const,
    /// Constant generic parameter: `struct Foo<const N: usize> { ... }`
    ConstParam,
    Static {
        /// Whether it's a `unsafe static`, `safe static` (inside extern only) or just a `static`.
        safety: hir::Safety,
        /// Whether it's a `static mut` or just a `static`.
        mutability: ast::Mutability,
        /// Whether it's an anonymous static generated for nested allocations.
        nested: bool,
    },
    /// Refers to the struct or enum variant's constructor.
    ///
    /// The reason `Ctor` exists in addition to [`DefKind::Struct`] and
    /// [`DefKind::Variant`] is because structs and enum variants exist
    /// in the *type* namespace, whereas struct and enum variant *constructors*
    /// exist in the *value* namespace.
    ///
    /// You may wonder why enum variants exist in the type namespace as opposed
    /// to the value namespace. Check out [RFC 2593] for intuition on why that is.
    ///
    /// [RFC 2593]: https://github.com/rust-lang/rfcs/pull/2593
    Ctor(CtorOf, CtorKind),
    /// Associated function: `impl MyStruct { fn associated() {} }`
    /// or `trait Foo { fn associated() {} }`
    AssocFn,
    /// Associated constant: `trait MyTrait { const ASSOC: usize; }`
    AssocConst,

    // Macro namespace
    Macro(MacroKind),

    // Not namespaced (or they are, but we don't treat them so)
    ExternCrate,
    Use,
    /// An `extern` block.
    ForeignMod,
    /// Anonymous constant, e.g. the `1 + 2` in `[u8; 1 + 2]`.
    ///
    /// Not all anon-consts are actually still relevant in the HIR. We lower
    /// trivial const-arguments directly to `hir::ConstArgKind::Path`, at which
    /// point the definition for the anon-const ends up unused and incomplete.
    ///
    /// We do not provide any a `Span` for the definition and pretty much all other
    /// queries also ICE when using this `DefId`. Given that the `DefId` of such
    /// constants should only be reachable by iterating all definitions of a
    /// given crate, you should not have to worry about this.
    AnonConst,
    /// An inline constant, e.g. `const { 1 + 2 }`
    InlineConst,
    /// Opaque type, aka `impl Trait`.
    OpaqueTy,
    /// A field in a struct, enum or union. e.g.
    /// - `bar` in `struct Foo { bar: u8 }`
    /// - `Foo::Bar::0` in `enum Foo { Bar(u8) }`
    Field,
    /// Lifetime parameter: the `'a` in `struct Foo<'a> { ... }`
    LifetimeParam,
    /// A use of `global_asm!`.
    GlobalAsm,
    Impl {
        of_trait: bool,
    },
    /// A closure, coroutine, or coroutine-closure.
    ///
    /// These are all represented with the same `ExprKind::Closure` in the AST and HIR,
    /// which makes it difficult to distinguish these during def collection. Therefore,
    /// we treat them all the same, and code which needs to distinguish them can match
    /// or `hir::ClosureKind` or `type_of`.
    Closure,
    /// The definition of a synthetic coroutine body created by the lowering of a
    /// coroutine-closure, such as an async closure.
    SyntheticCoroutineBody,
}

impl DefKind {
    /// Get an English description for the item's kind.
    ///
    /// If you have access to `TyCtxt`, use `TyCtxt::def_descr` or
    /// `TyCtxt::def_kind_descr` instead, because they give better
    /// information for coroutines and associated functions.
    pub fn descr(self, def_id: DefId) -> &'static str {
        match self {
            DefKind::Fn => "function",
            DefKind::Mod if def_id.is_crate_root() && !def_id.is_local() => "crate",
            DefKind::Mod => "module",
            DefKind::Static { .. } => "static",
            DefKind::Enum => "enum",
            DefKind::Variant => "variant",
            DefKind::Ctor(CtorOf::Variant, CtorKind::Fn) => "tuple variant",
            DefKind::Ctor(CtorOf::Variant, CtorKind::Const) => "unit variant",
            DefKind::Struct => "struct",
            DefKind::Ctor(CtorOf::Struct, CtorKind::Fn) => "tuple struct",
            DefKind::Ctor(CtorOf::Struct, CtorKind::Const) => "unit struct",
            DefKind::OpaqueTy => "opaque type",
            DefKind::TyAlias => "type alias",
            DefKind::TraitAlias => "trait alias",
            DefKind::AssocTy => "associated type",
            DefKind::Union => "union",
            DefKind::Trait => "trait",
            DefKind::ForeignTy => "foreign type",
            DefKind::AssocFn => "associated function",
            DefKind::Const => "constant",
            DefKind::AssocConst => "associated constant",
            DefKind::TyParam => "type parameter",
            DefKind::ConstParam => "const parameter",
            DefKind::Macro(macro_kind) => macro_kind.descr(),
            DefKind::LifetimeParam => "lifetime parameter",
            DefKind::Use => "import",
            DefKind::ForeignMod => "foreign module",
            DefKind::AnonConst => "constant expression",
            DefKind::InlineConst => "inline constant",
            DefKind::Field => "field",
            DefKind::Impl { .. } => "implementation",
            DefKind::Closure => "closure",
            DefKind::ExternCrate => "extern crate",
            DefKind::GlobalAsm => "global assembly block",
            DefKind::SyntheticCoroutineBody => "synthetic mir body",
        }
    }

    /// Gets an English article for the definition.
    ///
    /// If you have access to `TyCtxt`, use `TyCtxt::def_descr_article` or
    /// `TyCtxt::def_kind_descr_article` instead, because they give better
    /// information for coroutines and associated functions.
    pub fn article(&self) -> &'static str {
        match *self {
            DefKind::AssocTy
            | DefKind::AssocConst
            | DefKind::AssocFn
            | DefKind::Enum
            | DefKind::OpaqueTy
            | DefKind::Impl { .. }
            | DefKind::Use
            | DefKind::InlineConst
            | DefKind::ExternCrate => "an",
            DefKind::Macro(macro_kind) => macro_kind.article(),
            _ => "a",
        }
    }

    pub fn ns(&self) -> Option<Namespace> {
        match self {
            DefKind::Mod
            | DefKind::Struct
            | DefKind::Union
            | DefKind::Enum
            | DefKind::Variant
            | DefKind::Trait
            | DefKind::TyAlias
            | DefKind::ForeignTy
            | DefKind::TraitAlias
            | DefKind::AssocTy
            | DefKind::TyParam => Some(Namespace::TypeNS),

            DefKind::Fn
            | DefKind::Const
            | DefKind::ConstParam
            | DefKind::Static { .. }
            | DefKind::Ctor(..)
            | DefKind::AssocFn
            | DefKind::AssocConst => Some(Namespace::ValueNS),

            DefKind::Macro(..) => Some(Namespace::MacroNS),

            // Not namespaced.
            DefKind::AnonConst
            | DefKind::InlineConst
            | DefKind::Field
            | DefKind::LifetimeParam
            | DefKind::ExternCrate
            | DefKind::Closure
            | DefKind::Use
            | DefKind::ForeignMod
            | DefKind::GlobalAsm
            | DefKind::Impl { .. }
            | DefKind::OpaqueTy
            | DefKind::SyntheticCoroutineBody => None,
        }
    }

    // Some `DefKind`s require a name, some don't. Panics if one is needed but
    // not provided. (`AssocTy` is an exception, see below.)
    pub fn def_path_data(self, name: Option<Symbol>) -> DefPathData {
        match self {
            DefKind::Mod
            | DefKind::Struct
            | DefKind::Union
            | DefKind::Enum
            | DefKind::Variant
            | DefKind::Trait
            | DefKind::TyAlias
            | DefKind::ForeignTy
            | DefKind::TraitAlias
            | DefKind::TyParam
            | DefKind::ExternCrate => DefPathData::TypeNs(name.unwrap()),

            // An associated type name will be missing for an RPITIT (DefPathData::AnonAssocTy),
            // but those provide their own DefPathData.
            DefKind::AssocTy => DefPathData::TypeNs(name.unwrap()),

            DefKind::Fn
            | DefKind::Const
            | DefKind::ConstParam
            | DefKind::Static { .. }
            | DefKind::AssocFn
            | DefKind::AssocConst
            | DefKind::Field => DefPathData::ValueNs(name.unwrap()),
            DefKind::Macro(..) => DefPathData::MacroNs(name.unwrap()),
            DefKind::LifetimeParam => DefPathData::LifetimeNs(name.unwrap()),
            DefKind::Ctor(..) => DefPathData::Ctor,
            DefKind::Use => DefPathData::Use,
            DefKind::ForeignMod => DefPathData::ForeignMod,
            DefKind::AnonConst => DefPathData::AnonConst,
            DefKind::InlineConst => DefPathData::AnonConst,
            DefKind::OpaqueTy => DefPathData::OpaqueTy,
            DefKind::GlobalAsm => DefPathData::GlobalAsm,
            DefKind::Impl { .. } => DefPathData::Impl,
            DefKind::Closure => DefPathData::Closure,
            DefKind::SyntheticCoroutineBody => DefPathData::SyntheticCoroutineBody,
        }
    }

    #[inline]
    pub fn is_fn_like(self) -> bool {
        matches!(
            self,
            DefKind::Fn | DefKind::AssocFn | DefKind::Closure | DefKind::SyntheticCoroutineBody
        )
    }

    /// Whether `query get_codegen_attrs` should be used with this definition.
    pub fn has_codegen_attrs(self) -> bool {
        match self {
            DefKind::Fn
            | DefKind::AssocFn
            | DefKind::Ctor(..)
            | DefKind::Closure
            | DefKind::Static { .. }
            | DefKind::SyntheticCoroutineBody => true,
            DefKind::Mod
            | DefKind::Struct
            | DefKind::Union
            | DefKind::Enum
            | DefKind::Variant
            | DefKind::Trait
            | DefKind::TyAlias
            | DefKind::ForeignTy
            | DefKind::TraitAlias
            | DefKind::AssocTy
            | DefKind::Const
            | DefKind::AssocConst
            | DefKind::Macro(..)
            | DefKind::Use
            | DefKind::ForeignMod
            | DefKind::OpaqueTy
            | DefKind::Impl { .. }
            | DefKind::Field
            | DefKind::TyParam
            | DefKind::ConstParam
            | DefKind::LifetimeParam
            | DefKind::AnonConst
            | DefKind::InlineConst
            | DefKind::GlobalAsm
            | DefKind::ExternCrate => false,
        }
    }
}

/// The resolution of a path or export.
///
/// For every path or identifier in Rust, the compiler must determine
/// what the path refers to. This process is called name resolution,
/// and `Res` is the primary result of name resolution.
///
/// For example, everything prefixed with `/* Res */` in this example has
/// an associated `Res`:
///
/// ```
/// fn str_to_string(s: & /* Res */ str) -> /* Res */ String {
///     /* Res */ String::from(/* Res */ s)
/// }
///
/// /* Res */ str_to_string("hello");
/// ```
///
/// The associated `Res`s will be:
///
/// - `str` will resolve to [`Res::PrimTy`];
/// - `String` will resolve to [`Res::Def`], and the `Res` will include the [`DefId`]
///   for `String` as defined in the standard library;
/// - `String::from` will also resolve to [`Res::Def`], with the [`DefId`]
///   pointing to `String::from`;
/// - `s` will resolve to [`Res::Local`];
/// - the call to `str_to_string` will resolve to [`Res::Def`], with the [`DefId`]
///   pointing to the definition of `str_to_string` in the current crate.
//
#[derive(Clone, Copy, PartialEq, Eq, Encodable, Decodable, Hash, Debug, HashStable_Generic)]
pub enum Res<Id = hir::HirId> {
    /// Definition having a unique ID (`DefId`), corresponds to something defined in user code.
    ///
    /// **Not bound to a specific namespace.**
    Def(DefKind, DefId),

    // Type namespace
    /// A primitive type such as `i32` or `str`.
    ///
    /// **Belongs to the type namespace.**
    PrimTy(hir::PrimTy),

    /// The `Self` type, as used within a trait.
    ///
    /// **Belongs to the type namespace.**
    ///
    /// See the examples on [`Res::SelfTyAlias`] for details.
    SelfTyParam {
        /// The trait this `Self` is a generic parameter for.
        trait_: DefId,
    },

    /// The `Self` type, as used somewhere other than within a trait.
    ///
    /// **Belongs to the type namespace.**
    ///
    /// Examples:
    /// ```
    /// struct Bar(Box<Self>); // SelfTyAlias
    ///
    /// trait Foo {
    ///     fn foo() -> Box<Self>; // SelfTyParam
    /// }
    ///
    /// impl Bar {
    ///     fn blah() {
    ///         let _: Self; // SelfTyAlias
    ///     }
    /// }
    ///
    /// impl Foo for Bar {
    ///     fn foo() -> Box<Self> { // SelfTyAlias
    ///         let _: Self;        // SelfTyAlias
    ///
    ///         todo!()
    ///     }
    /// }
    /// ```
    /// *See also [`Res::SelfCtor`].*
    ///
    SelfTyAlias {
        /// The item introducing the `Self` type alias. Can be used in the `type_of` query
        /// to get the underlying type.
        alias_to: DefId,

        /// Whether the `Self` type is disallowed from mentioning generics (i.e. when used in an
        /// anonymous constant).
        ///
        /// HACK(min_const_generics): self types also have an optional requirement to **not**
        /// mention any generic parameters to allow the following with `min_const_generics`:
        /// ```
        /// # struct Foo;
        /// impl Foo { fn test() -> [u8; size_of::<Self>()] { todo!() } }
        ///
        /// struct Bar([u8; baz::<Self>()]);
        /// const fn baz<T>() -> usize { 10 }
        /// ```
        /// We do however allow `Self` in repeat expression even if it is generic to not break code
        /// which already works on stable while causing the `const_evaluatable_unchecked` future
        /// compat lint:
        /// ```
        /// fn foo<T>() {
        ///     let _bar = [1_u8; size_of::<*mut T>()];
        /// }
        /// ```
        // FIXME(generic_const_exprs): Remove this bodge once that feature is stable.
        forbid_generic: bool,

        /// Is this within an `impl Foo for bar`?
        is_trait_impl: bool,
    },

    // Value namespace
    /// The `Self` constructor, along with the [`DefId`]
    /// of the impl it is associated with.
    ///
    /// **Belongs to the value namespace.**
    ///
    /// *See also [`Res::SelfTyParam`] and [`Res::SelfTyAlias`].*
    SelfCtor(DefId),

    /// A local variable or function parameter.
    ///
    /// **Belongs to the value namespace.**
    Local(Id),

    /// A tool attribute module; e.g., the `rustfmt` in `#[rustfmt::skip]`.
    ///
    /// **Belongs to the type namespace.**
    ToolMod,

    // Macro namespace
    /// An attribute that is *not* implemented via macro.
    /// E.g., `#[inline]` and `#[rustfmt::skip]`, which are essentially directives,
    /// as opposed to `#[test]`, which is a builtin macro.
    ///
    /// **Belongs to the macro namespace.**
    NonMacroAttr(NonMacroAttrKind), // e.g., `#[inline]` or `#[rustfmt::skip]`

    // All namespaces
    /// Name resolution failed. We use a dummy `Res` variant so later phases
    /// of the compiler won't crash and can instead report more errors.
    ///
    /// **Not bound to a specific namespace.**
    Err,
}

/// The result of resolving a path before lowering to HIR,
/// with "module" segments resolved and associated item
/// segments deferred to type checking.
/// `base_res` is the resolution of the resolved part of the
/// path, `unresolved_segments` is the number of unresolved
/// segments.
///
/// ```text
/// module::Type::AssocX::AssocY::MethodOrAssocType
/// ^~~~~~~~~~~~  ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// base_res      unresolved_segments = 3
///
/// <T as Trait>::AssocX::AssocY::MethodOrAssocType
///       ^~~~~~~~~~~~~~  ^~~~~~~~~~~~~~~~~~~~~~~~~
///       base_res        unresolved_segments = 2
/// ```
#[derive(Copy, Clone, Debug)]
pub struct PartialRes {
    base_res: Res<NodeId>,
    unresolved_segments: usize,
}

impl PartialRes {
    #[inline]
    pub fn new(base_res: Res<NodeId>) -> Self {
        PartialRes { base_res, unresolved_segments: 0 }
    }

    #[inline]
    pub fn with_unresolved_segments(base_res: Res<NodeId>, mut unresolved_segments: usize) -> Self {
        if base_res == Res::Err {
            unresolved_segments = 0
        }
        PartialRes { base_res, unresolved_segments }
    }

    #[inline]
    pub fn base_res(&self) -> Res<NodeId> {
        self.base_res
    }

    #[inline]
    pub fn unresolved_segments(&self) -> usize {
        self.unresolved_segments
    }

    #[inline]
    pub fn full_res(&self) -> Option<Res<NodeId>> {
        (self.unresolved_segments == 0).then_some(self.base_res)
    }

    #[inline]
    pub fn expect_full_res(&self) -> Res<NodeId> {
        self.full_res().expect("unexpected unresolved segments")
    }
}

/// Different kinds of symbols can coexist even if they share the same textual name.
/// Therefore, they each have a separate universe (known as a "namespace").
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Encodable, Decodable)]
#[derive(HashStable_Generic)]
pub enum Namespace {
    /// The type namespace includes `struct`s, `enum`s, `union`s, `trait`s, and `mod`s
    /// (and, by extension, crates).
    ///
    /// Note that the type namespace includes other items; this is not an
    /// exhaustive list.
    TypeNS,
    /// The value namespace includes `fn`s, `const`s, `static`s, and local variables (including function arguments).
    ValueNS,
    /// The macro namespace includes `macro_rules!` macros, declarative `macro`s,
    /// procedural macros, attribute macros, `derive` macros, and non-macro attributes
    /// like `#[inline]` and `#[rustfmt::skip]`.
    MacroNS,
}

impl Namespace {
    /// The English description of the namespace.
    pub fn descr(self) -> &'static str {
        match self {
            Self::TypeNS => "type",
            Self::ValueNS => "value",
            Self::MacroNS => "macro",
        }
    }
}

impl<CTX: crate::HashStableContext> ToStableHashKey<CTX> for Namespace {
    type KeyType = Namespace;

    #[inline]
    fn to_stable_hash_key(&self, _: &CTX) -> Namespace {
        *self
    }
}

/// Just a helper ‒ separate structure for each namespace.
#[derive(Copy, Clone, Default, Debug, HashStable_Generic)]
pub struct PerNS<T> {
    pub value_ns: T,
    pub type_ns: T,
    pub macro_ns: T,
}

impl<T> PerNS<T> {
    pub fn map<U, F: FnMut(T) -> U>(self, mut f: F) -> PerNS<U> {
        PerNS { value_ns: f(self.value_ns), type_ns: f(self.type_ns), macro_ns: f(self.macro_ns) }
    }

    /// Note: Do you really want to use this? Often you know which namespace a
    /// name will belong in, and you can consider just that namespace directly,
    /// rather than iterating through all of them.
    pub fn into_iter(self) -> IntoIter<T, 3> {
        [self.value_ns, self.type_ns, self.macro_ns].into_iter()
    }

    /// Note: Do you really want to use this? Often you know which namespace a
    /// name will belong in, and you can consider just that namespace directly,
    /// rather than iterating through all of them.
    pub fn iter(&self) -> IntoIter<&T, 3> {
        [&self.value_ns, &self.type_ns, &self.macro_ns].into_iter()
    }
}

impl<T> ::std::ops::Index<Namespace> for PerNS<T> {
    type Output = T;

    fn index(&self, ns: Namespace) -> &T {
        match ns {
            Namespace::ValueNS => &self.value_ns,
            Namespace::TypeNS => &self.type_ns,
            Namespace::MacroNS => &self.macro_ns,
        }
    }
}

impl<T> ::std::ops::IndexMut<Namespace> for PerNS<T> {
    fn index_mut(&mut self, ns: Namespace) -> &mut T {
        match ns {
            Namespace::ValueNS => &mut self.value_ns,
            Namespace::TypeNS => &mut self.type_ns,
            Namespace::MacroNS => &mut self.macro_ns,
        }
    }
}

impl<T> PerNS<Option<T>> {
    /// Returns `true` if all the items in this collection are `None`.
    pub fn is_empty(&self) -> bool {
        self.type_ns.is_none() && self.value_ns.is_none() && self.macro_ns.is_none()
    }

    /// Returns an iterator over the items which are `Some`.
    ///
    /// Note: Do you really want to use this? Often you know which namespace a
    /// name will belong in, and you can consider just that namespace directly,
    /// rather than iterating through all of them.
    pub fn present_items(self) -> impl Iterator<Item = T> {
        [self.type_ns, self.value_ns, self.macro_ns].into_iter().flatten()
    }
}

impl CtorKind {
    pub fn from_ast(vdata: &ast::VariantData) -> Option<(CtorKind, NodeId)> {
        match *vdata {
            ast::VariantData::Tuple(_, node_id) => Some((CtorKind::Fn, node_id)),
            ast::VariantData::Unit(node_id) => Some((CtorKind::Const, node_id)),
            ast::VariantData::Struct { .. } => None,
        }
    }
}

impl NonMacroAttrKind {
    pub fn descr(self) -> &'static str {
        match self {
            NonMacroAttrKind::Builtin(..) => "built-in attribute",
            NonMacroAttrKind::Tool => "tool attribute",
            NonMacroAttrKind::DeriveHelper | NonMacroAttrKind::DeriveHelperCompat => {
                "derive helper attribute"
            }
        }
    }

    // Currently trivial, but exists in case a new kind is added in the future whose name starts
    // with a vowel.
    pub fn article(self) -> &'static str {
        "a"
    }

    /// Users of some attributes cannot mark them as used, so they are considered always used.
    pub fn is_used(self) -> bool {
        match self {
            NonMacroAttrKind::Tool
            | NonMacroAttrKind::DeriveHelper
            | NonMacroAttrKind::DeriveHelperCompat => true,
            NonMacroAttrKind::Builtin(..) => false,
        }
    }
}

impl<Id> Res<Id> {
    /// Return the `DefId` of this `Def` if it has an ID, else panic.
    pub fn def_id(&self) -> DefId
    where
        Id: Debug,
    {
        self.opt_def_id().unwrap_or_else(|| panic!("attempted .def_id() on invalid res: {self:?}"))
    }

    /// Return `Some(..)` with the `DefId` of this `Res` if it has a ID, else `None`.
    pub fn opt_def_id(&self) -> Option<DefId> {
        match *self {
            Res::Def(_, id) => Some(id),

            Res::Local(..)
            | Res::PrimTy(..)
            | Res::SelfTyParam { .. }
            | Res::SelfTyAlias { .. }
            | Res::SelfCtor(..)
            | Res::ToolMod
            | Res::NonMacroAttr(..)
            | Res::Err => None,
        }
    }

    /// Return the `DefId` of this `Res` if it represents a module.
    pub fn mod_def_id(&self) -> Option<DefId> {
        match *self {
            Res::Def(DefKind::Mod, id) => Some(id),
            _ => None,
        }
    }

    /// A human readable name for the res kind ("function", "module", etc.).
    pub fn descr(&self) -> &'static str {
        match *self {
            Res::Def(kind, def_id) => kind.descr(def_id),
            Res::SelfCtor(..) => "self constructor",
            Res::PrimTy(..) => "builtin type",
            Res::Local(..) => "local variable",
            Res::SelfTyParam { .. } | Res::SelfTyAlias { .. } => "self type",
            Res::ToolMod => "tool module",
            Res::NonMacroAttr(attr_kind) => attr_kind.descr(),
            Res::Err => "unresolved item",
        }
    }

    /// Gets an English article for the `Res`.
    pub fn article(&self) -> &'static str {
        match *self {
            Res::Def(kind, _) => kind.article(),
            Res::NonMacroAttr(kind) => kind.article(),
            Res::Err => "an",
            _ => "a",
        }
    }

    pub fn map_id<R>(self, mut map: impl FnMut(Id) -> R) -> Res<R> {
        match self {
            Res::Def(kind, id) => Res::Def(kind, id),
            Res::SelfCtor(id) => Res::SelfCtor(id),
            Res::PrimTy(id) => Res::PrimTy(id),
            Res::Local(id) => Res::Local(map(id)),
            Res::SelfTyParam { trait_ } => Res::SelfTyParam { trait_ },
            Res::SelfTyAlias { alias_to, forbid_generic, is_trait_impl } => {
                Res::SelfTyAlias { alias_to, forbid_generic, is_trait_impl }
            }
            Res::ToolMod => Res::ToolMod,
            Res::NonMacroAttr(attr_kind) => Res::NonMacroAttr(attr_kind),
            Res::Err => Res::Err,
        }
    }

    pub fn apply_id<R, E>(self, mut map: impl FnMut(Id) -> Result<R, E>) -> Result<Res<R>, E> {
        Ok(match self {
            Res::Def(kind, id) => Res::Def(kind, id),
            Res::SelfCtor(id) => Res::SelfCtor(id),
            Res::PrimTy(id) => Res::PrimTy(id),
            Res::Local(id) => Res::Local(map(id)?),
            Res::SelfTyParam { trait_ } => Res::SelfTyParam { trait_ },
            Res::SelfTyAlias { alias_to, forbid_generic, is_trait_impl } => {
                Res::SelfTyAlias { alias_to, forbid_generic, is_trait_impl }
            }
            Res::ToolMod => Res::ToolMod,
            Res::NonMacroAttr(attr_kind) => Res::NonMacroAttr(attr_kind),
            Res::Err => Res::Err,
        })
    }

    #[track_caller]
    pub fn expect_non_local<OtherId>(self) -> Res<OtherId> {
        self.map_id(
            #[track_caller]
            |_| panic!("unexpected `Res::Local`"),
        )
    }

    pub fn macro_kind(self) -> Option<MacroKind> {
        match self {
            Res::Def(DefKind::Macro(kind), _) => Some(kind),
            Res::NonMacroAttr(..) => Some(MacroKind::Attr),
            _ => None,
        }
    }

    /// Returns `None` if this is `Res::Err`
    pub fn ns(&self) -> Option<Namespace> {
        match self {
            Res::Def(kind, ..) => kind.ns(),
            Res::PrimTy(..) | Res::SelfTyParam { .. } | Res::SelfTyAlias { .. } | Res::ToolMod => {
                Some(Namespace::TypeNS)
            }
            Res::SelfCtor(..) | Res::Local(..) => Some(Namespace::ValueNS),
            Res::NonMacroAttr(..) => Some(Namespace::MacroNS),
            Res::Err => None,
        }
    }

    /// Always returns `true` if `self` is `Res::Err`
    pub fn matches_ns(&self, ns: Namespace) -> bool {
        self.ns().is_none_or(|actual_ns| actual_ns == ns)
    }

    /// Returns whether such a resolved path can occur in a tuple struct/variant pattern
    pub fn expected_in_tuple_struct_pat(&self) -> bool {
        matches!(self, Res::Def(DefKind::Ctor(_, CtorKind::Fn), _) | Res::SelfCtor(..))
    }

    /// Returns whether such a resolved path can occur in a unit struct/variant pattern
    pub fn expected_in_unit_struct_pat(&self) -> bool {
        matches!(self, Res::Def(DefKind::Ctor(_, CtorKind::Const), _) | Res::SelfCtor(..))
    }
}

/// Resolution for a lifetime appearing in a type.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum LifetimeRes {
    /// Successfully linked the lifetime to a generic parameter.
    Param {
        /// Id of the generic parameter that introduced it.
        param: LocalDefId,
        /// Id of the introducing place. That can be:
        /// - an item's id, for the item's generic parameters;
        /// - a TraitRef's ref_id, identifying the `for<...>` binder;
        /// - a BareFn type's id.
        ///
        /// This information is used for impl-trait lifetime captures, to know when to or not to
        /// capture any given lifetime.
        binder: NodeId,
    },
    /// Created a generic parameter for an anonymous lifetime.
    Fresh {
        /// Id of the generic parameter that introduced it.
        ///
        /// Creating the associated `LocalDefId` is the responsibility of lowering.
        param: NodeId,
        /// Id of the introducing place. See `Param`.
        binder: NodeId,
        /// Kind of elided lifetime
        kind: hir::MissingLifetimeKind,
    },
    /// This variant is used for anonymous lifetimes that we did not resolve during
    /// late resolution. Those lifetimes will be inferred by typechecking.
    Infer,
    /// `'static` lifetime.
    Static {
        /// We do not want to emit `elided_named_lifetimes`
        /// when we are inside of a const item or a static,
        /// because it would get too annoying.
        suppress_elision_warning: bool,
    },
    /// Resolution failure.
    Error,
    /// HACK: This is used to recover the NodeId of an elided lifetime.
    ElidedAnchor { start: NodeId, end: NodeId },
}

pub type DocLinkResMap = UnordMap<(Symbol, Namespace), Option<Res<NodeId>>>;
