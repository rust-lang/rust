use crate::hir::def_id::DefId;
use crate::util::nodemap::DefIdMap;
use syntax::ast;
use syntax::ext::base::MacroKind;
use syntax::ast::NodeId;
use syntax_pos::Span;
use rustc_macros::HashStable;
use crate::hir;
use crate::ty;
use std::fmt::Debug;

use self::Namespace::*;

/// Encodes if a `DefKind::Ctor` is the constructor of an enum variant or a struct.
#[derive(Clone, Copy, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, HashStable)]
pub enum CtorOf {
    /// This `DefKind::Ctor` is a synthesized constructor of a tuple or unit struct.
    Struct,
    /// This `DefKind::Ctor` is a synthesized constructor of a tuple or unit variant.
    Variant,
}

#[derive(Clone, Copy, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, HashStable)]
pub enum CtorKind {
    /// Constructor function automatically created by a tuple struct/variant.
    Fn,
    /// Constructor constant automatically created by a unit struct/variant.
    Const,
    /// Unusable name in value namespace created by a struct variant.
    Fictive,
}

#[derive(Clone, Copy, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, HashStable)]
pub enum NonMacroAttrKind {
    /// Single-segment attribute defined by the language (`#[inline]`)
    Builtin,
    /// Multi-segment custom attribute living in a "tool module" (`#[rustfmt::skip]`).
    Tool,
    /// Single-segment custom attribute registered by a derive macro (`#[serde(default)]`).
    DeriveHelper,
    /// Single-segment custom attribute registered by a legacy plugin (`register_attribute`).
    LegacyPluginHelper,
    /// Single-segment custom attribute not registered in any way (`#[my_attr]`).
    Custom,
}

#[derive(Clone, Copy, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, HashStable)]
pub enum DefKind {
    // Type namespace
    Mod,
    /// Refers to the struct itself, `DefKind::Ctor` refers to its constructor if it exists.
    Struct,
    Union,
    Enum,
    /// Refers to the variant itself, `DefKind::Ctor` refers to its constructor if it exists.
    Variant,
    Trait,
    /// `existential type Foo: Bar;`
    Existential,
    /// `type Foo = Bar;`
    TyAlias,
    ForeignTy,
    TraitAlias,
    AssocTy,
    /// `existential type Foo: Bar;`
    AssocExistential,
    TyParam,

    // Value namespace
    Fn,
    Const,
    ConstParam,
    Static,
    /// Refers to the struct or enum variant's constructor.
    Ctor(CtorOf, CtorKind),
    Method,
    AssocConst,

    // Macro namespace
    Macro(MacroKind),
}

impl DefKind {
    pub fn descr(self) -> &'static str {
        match self {
            DefKind::Fn => "function",
            DefKind::Mod => "module",
            DefKind::Static => "static",
            DefKind::Enum => "enum",
            DefKind::Variant => "variant",
            DefKind::Ctor(CtorOf::Variant, CtorKind::Fn) => "tuple variant",
            DefKind::Ctor(CtorOf::Variant, CtorKind::Const) => "unit variant",
            DefKind::Ctor(CtorOf::Variant, CtorKind::Fictive) => "struct variant",
            DefKind::Struct => "struct",
            DefKind::Ctor(CtorOf::Struct, CtorKind::Fn) => "tuple struct",
            DefKind::Ctor(CtorOf::Struct, CtorKind::Const) => "unit struct",
            DefKind::Ctor(CtorOf::Struct, CtorKind::Fictive) =>
                bug!("impossible struct constructor"),
            DefKind::Existential => "existential type",
            DefKind::TyAlias => "type alias",
            DefKind::TraitAlias => "trait alias",
            DefKind::AssocTy => "associated type",
            DefKind::AssocExistential => "associated existential type",
            DefKind::Union => "union",
            DefKind::Trait => "trait",
            DefKind::ForeignTy => "foreign type",
            DefKind::Method => "method",
            DefKind::Const => "constant",
            DefKind::AssocConst => "associated constant",
            DefKind::TyParam => "type parameter",
            DefKind::ConstParam => "const parameter",
            DefKind::Macro(macro_kind) => macro_kind.descr(),
        }
    }

    /// An English article for the def.
    pub fn article(&self) -> &'static str {
        match *self {
            DefKind::AssocTy
            | DefKind::AssocConst
            | DefKind::AssocExistential
            | DefKind::Enum
            | DefKind::Existential => "an",
            DefKind::Macro(macro_kind) => macro_kind.article(),
            _ => "a",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, HashStable)]
pub enum Res<Id = hir::HirId> {
    Def(DefKind, DefId),

    // Type namespace
    PrimTy(hir::PrimTy),
    SelfTy(Option<DefId> /* trait */, Option<DefId> /* impl */),
    ToolMod, // e.g., `rustfmt` in `#[rustfmt::skip]`

    // Value namespace
    SelfCtor(DefId /* impl */),  // `DefId` refers to the impl
    Local(Id),

    // Macro namespace
    NonMacroAttr(NonMacroAttrKind), // e.g., `#[inline]` or `#[rustfmt::skip]`

    // All namespaces
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
        if base_res == Res::Err { unresolved_segments = 0 }
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
}

/// Different kinds of symbols don't influence each other.
///
/// Therefore, they have a separate universe (namespace).
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Namespace {
    TypeNS,
    ValueNS,
    MacroNS,
}

impl Namespace {
    pub fn descr(self) -> &'static str {
        match self {
            TypeNS => "type",
            ValueNS => "value",
            MacroNS => "macro",
        }
    }
}

/// Just a helper ‒ separate structure for each namespace.
#[derive(Copy, Clone, Default, Debug)]
pub struct PerNS<T> {
    pub value_ns: T,
    pub type_ns: T,
    pub macro_ns: T,
}

impl<T> PerNS<T> {
    pub fn map<U, F: FnMut(T) -> U>(self, mut f: F) -> PerNS<U> {
        PerNS {
            value_ns: f(self.value_ns),
            type_ns: f(self.type_ns),
            macro_ns: f(self.macro_ns),
        }
    }
}

impl<T> ::std::ops::Index<Namespace> for PerNS<T> {
    type Output = T;

    fn index(&self, ns: Namespace) -> &T {
        match ns {
            ValueNS => &self.value_ns,
            TypeNS => &self.type_ns,
            MacroNS => &self.macro_ns,
        }
    }
}

impl<T> ::std::ops::IndexMut<Namespace> for PerNS<T> {
    fn index_mut(&mut self, ns: Namespace) -> &mut T {
        match ns {
            ValueNS => &mut self.value_ns,
            TypeNS => &mut self.type_ns,
            MacroNS => &mut self.macro_ns,
        }
    }
}

impl<T> PerNS<Option<T>> {
    /// Returns `true` if all the items in this collection are `None`.
    pub fn is_empty(&self) -> bool {
        self.type_ns.is_none() && self.value_ns.is_none() && self.macro_ns.is_none()
    }

    /// Returns an iterator over the items which are `Some`.
    pub fn present_items(self) -> impl Iterator<Item=T> {
        use std::iter::once;

        once(self.type_ns)
            .chain(once(self.value_ns))
            .chain(once(self.macro_ns))
            .filter_map(|it| it)
    }
}

/// This is the replacement export map. It maps a module to all of the exports
/// within.
pub type ExportMap<Id> = DefIdMap<Vec<Export<Id>>>;

#[derive(Copy, Clone, Debug, RustcEncodable, RustcDecodable, HashStable)]
pub struct Export<Id> {
    /// The name of the target.
    pub ident: ast::Ident,
    /// The resolution of the target.
    pub res: Res<Id>,
    /// The span of the target.
    pub span: Span,
    /// The visibility of the export.
    /// We include non-`pub` exports for hygienic macros that get used from extern crates.
    pub vis: ty::Visibility,
}

impl<Id> Export<Id> {
    pub fn map_id<R>(self, map: impl FnMut(Id) -> R) -> Export<R> {
        Export {
            ident: self.ident,
            res: self.res.map_id(map),
            span: self.span,
            vis: self.vis,
        }
    }
}

impl CtorKind {
    pub fn from_ast(vdata: &ast::VariantData) -> CtorKind {
        match *vdata {
            ast::VariantData::Tuple(..) => CtorKind::Fn,
            ast::VariantData::Unit(..) => CtorKind::Const,
            ast::VariantData::Struct(..) => CtorKind::Fictive,
        }
    }

    pub fn from_hir(vdata: &hir::VariantData) -> CtorKind {
        match *vdata {
            hir::VariantData::Tuple(..) => CtorKind::Fn,
            hir::VariantData::Unit(..) => CtorKind::Const,
            hir::VariantData::Struct(..) => CtorKind::Fictive,
        }
    }
}

impl NonMacroAttrKind {
    pub fn descr(self) -> &'static str {
        match self {
            NonMacroAttrKind::Builtin => "built-in attribute",
            NonMacroAttrKind::Tool => "tool attribute",
            NonMacroAttrKind::DeriveHelper => "derive helper attribute",
            NonMacroAttrKind::LegacyPluginHelper => "legacy plugin helper attribute",
            NonMacroAttrKind::Custom => "custom attribute",
        }
    }
}

impl<Id> Res<Id> {
    /// Return the `DefId` of this `Def` if it has an id, else panic.
    pub fn def_id(&self) -> DefId
    where
        Id: Debug,
    {
        self.opt_def_id().unwrap_or_else(|| {
            bug!("attempted .def_id() on invalid res: {:?}", self)
        })
    }

    /// Return `Some(..)` with the `DefId` of this `Res` if it has a id, else `None`.
    pub fn opt_def_id(&self) -> Option<DefId> {
        match *self {
            Res::Def(_, id) => Some(id),

            Res::Local(..) |
            Res::PrimTy(..) |
            Res::SelfTy(..) |
            Res::SelfCtor(..) |
            Res::ToolMod |
            Res::NonMacroAttr(..) |
            Res::Err => {
                None
            }
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
            Res::Def(kind, _) => kind.descr(),
            Res::SelfCtor(..) => "self constructor",
            Res::PrimTy(..) => "builtin type",
            Res::Local(..) => "local variable",
            Res::SelfTy(..) => "self type",
            Res::ToolMod => "tool module",
            Res::NonMacroAttr(attr_kind) => attr_kind.descr(),
            Res::Err => "unresolved item",
        }
    }

    /// An English article for the res.
    pub fn article(&self) -> &'static str {
        match *self {
            Res::Def(kind, _) => kind.article(),
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
            Res::SelfTy(a, b) => Res::SelfTy(a, b),
            Res::ToolMod => Res::ToolMod,
            Res::NonMacroAttr(attr_kind) => Res::NonMacroAttr(attr_kind),
            Res::Err => Res::Err,
        }
    }
}
