use crate::hir::def_id::DefId;
use crate::util::nodemap::{NodeMap, DefIdMap};
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
    AssociatedTy,
    /// `existential type Foo: Bar;`
    AssociatedExistential,
    TyParam,

    // Value namespace
    Fn,
    Const,
    ConstParam,
    Static,
    /// Refers to the struct or enum variant's constructor.
    Ctor(CtorOf, CtorKind),
    Method,
    AssociatedConst,

    // Macro namespace
    Macro(MacroKind),
}

#[derive(Clone, Copy, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, HashStable)]
pub enum Def<Id = hir::HirId> {
    Def(DefKind, DefId),

    // Type namespace
    PrimTy(hir::PrimTy),
    SelfTy(Option<DefId> /* trait */, Option<DefId> /* impl */),
    ToolMod, // e.g., `rustfmt` in `#[rustfmt::skip]`

    // Value namespace
    SelfCtor(DefId /* impl */),  // `DefId` refers to the impl
    Local(Id),
    Upvar(Id,           // `HirId` of closed over local
          usize,        // index in the `freevars` list of the closure
          ast::NodeId), // expr node that creates the closure
    Label(ast::NodeId),

    // Macro namespace
    NonMacroAttr(NonMacroAttrKind), // e.g., `#[inline]` or `#[rustfmt::skip]`

    // All namespaces
    Err,
}

/// The result of resolving a path before lowering to HIR.
/// `base_def` is definition of resolved part of the
/// path, `unresolved_segments` is the number of unresolved
/// segments.
///
/// ```text
/// module::Type::AssocX::AssocY::MethodOrAssocType
/// ^~~~~~~~~~~~  ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// base_def      unresolved_segments = 3
///
/// <T as Trait>::AssocX::AssocY::MethodOrAssocType
///       ^~~~~~~~~~~~~~  ^~~~~~~~~~~~~~~~~~~~~~~~~
///       base_def        unresolved_segments = 2
/// ```
#[derive(Copy, Clone, Debug)]
pub struct PathResolution {
    base_def: Def<NodeId>,
    unresolved_segments: usize,
}

impl PathResolution {
    pub fn new(def: Def<NodeId>) -> Self {
        PathResolution { base_def: def, unresolved_segments: 0 }
    }

    pub fn with_unresolved_segments(def: Def<NodeId>, mut unresolved_segments: usize) -> Self {
        if def == Def::Err { unresolved_segments = 0 }
        PathResolution { base_def: def, unresolved_segments: unresolved_segments }
    }

    #[inline]
    pub fn base_def(&self) -> Def<NodeId> {
        self.base_def
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

/// Definition mapping
pub type DefMap = NodeMap<PathResolution>;

/// This is the replacement export map. It maps a module to all of the exports
/// within.
pub type ExportMap<Id> = DefIdMap<Vec<Export<Id>>>;

/// Map used to track the `use` statements within a scope, matching it with all the items in every
/// namespace.
pub type ImportMap = NodeMap<PerNS<Option<PathResolution>>>;

#[derive(Copy, Clone, Debug, RustcEncodable, RustcDecodable, HashStable)]
pub struct Export<Id> {
    /// The name of the target.
    pub ident: ast::Ident,
    /// The definition of the target.
    pub def: Def<Id>,
    /// The span of the target definition.
    pub span: Span,
    /// The visibility of the export.
    /// We include non-`pub` exports for hygienic macros that get used from extern crates.
    pub vis: ty::Visibility,
}

impl<Id> Export<Id> {
    pub fn map_id<R>(self, map: impl FnMut(Id) -> R) -> Export<R> {
        Export {
            ident: self.ident,
            def: self.def.map_id(map),
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

impl<Id> Def<Id> {
    /// Return the `DefId` of this `Def` if it has an id, else panic.
    pub fn def_id(&self) -> DefId
    where
        Id: Debug,
    {
        self.opt_def_id().unwrap_or_else(|| {
            bug!("attempted .def_id() on invalid def: {:?}", self)
        })
    }

    /// Return `Some(..)` with the `DefId` of this `Def` if it has a id, else `None`.
    pub fn opt_def_id(&self) -> Option<DefId> {
        match *self {
            Def::Def(_, id) => Some(id),

            Def::Local(..) |
            Def::Upvar(..) |
            Def::Label(..)  |
            Def::PrimTy(..) |
            Def::SelfTy(..) |
            Def::SelfCtor(..) |
            Def::ToolMod |
            Def::NonMacroAttr(..) |
            Def::Err => {
                None
            }
        }
    }

    /// Return the `DefId` of this `Def` if it represents a module.
    pub fn mod_def_id(&self) -> Option<DefId> {
        match *self {
            Def::Def(DefKind::Mod, id) => Some(id),
            _ => None,
        }
    }

    /// A human readable name for the def kind ("function", "module", etc.).
    pub fn kind_name(&self) -> &'static str {
        match *self {
            Def::Def(DefKind::Fn, _) => "function",
            Def::Def(DefKind::Mod, _) => "module",
            Def::Def(DefKind::Static, _) => "static",
            Def::Def(DefKind::Enum, _) => "enum",
            Def::Def(DefKind::Variant, _) => "variant",
            Def::Def(DefKind::Ctor(CtorOf::Variant, CtorKind::Fn), _) => "tuple variant",
            Def::Def(DefKind::Ctor(CtorOf::Variant, CtorKind::Const), _) => "unit variant",
            Def::Def(DefKind::Ctor(CtorOf::Variant, CtorKind::Fictive), _) => "struct variant",
            Def::Def(DefKind::Struct, _) => "struct",
            Def::Def(DefKind::Ctor(CtorOf::Struct, CtorKind::Fn), _) => "tuple struct",
            Def::Def(DefKind::Ctor(CtorOf::Struct, CtorKind::Const), _) => "unit struct",
            Def::Def(DefKind::Ctor(CtorOf::Struct, CtorKind::Fictive), _) =>
                bug!("impossible struct constructor"),
            Def::Def(DefKind::Existential, _) => "existential type",
            Def::Def(DefKind::TyAlias, _) => "type alias",
            Def::Def(DefKind::TraitAlias, _) => "trait alias",
            Def::Def(DefKind::AssociatedTy, _) => "associated type",
            Def::Def(DefKind::AssociatedExistential, _) => "associated existential type",
            Def::SelfCtor(..) => "self constructor",
            Def::Def(DefKind::Union, _) => "union",
            Def::Def(DefKind::Trait, _) => "trait",
            Def::Def(DefKind::ForeignTy, _) => "foreign type",
            Def::Def(DefKind::Method, _) => "method",
            Def::Def(DefKind::Const, _) => "constant",
            Def::Def(DefKind::AssociatedConst, _) => "associated constant",
            Def::Def(DefKind::TyParam, _) => "type parameter",
            Def::Def(DefKind::ConstParam, _) => "const parameter",
            Def::PrimTy(..) => "builtin type",
            Def::Local(..) => "local variable",
            Def::Upvar(..) => "closure capture",
            Def::Label(..) => "label",
            Def::SelfTy(..) => "self type",
            Def::Def(DefKind::Macro(macro_kind), _) => macro_kind.descr(),
            Def::ToolMod => "tool module",
            Def::NonMacroAttr(attr_kind) => attr_kind.descr(),
            Def::Err => "unresolved item",
        }
    }

    /// An English article for the def.
    pub fn article(&self) -> &'static str {
        match *self {
            Def::Def(DefKind::AssociatedTy, _)
            | Def::Def(DefKind::AssociatedConst, _)
            | Def::Def(DefKind::AssociatedExistential, _)
            | Def::Def(DefKind::Enum, _)
            | Def::Def(DefKind::Existential, _)
            | Def::Err => "an",
            Def::Def(DefKind::Macro(macro_kind), _) => macro_kind.article(),
            _ => "a",
        }
    }

    pub fn map_id<R>(self, mut map: impl FnMut(Id) -> R) -> Def<R> {
        match self {
            Def::Def(kind, id) => Def::Def(kind, id),
            Def::SelfCtor(id) => Def::SelfCtor(id),
            Def::PrimTy(id) => Def::PrimTy(id),
            Def::Local(id) => Def::Local(map(id)),
            Def::Upvar(id, index, closure) => Def::Upvar(
                map(id),
                index,
                closure
            ),
            Def::Label(id) => Def::Label(id),
            Def::SelfTy(a, b) => Def::SelfTy(a, b),
            Def::ToolMod => Def::ToolMod,
            Def::NonMacroAttr(attr_kind) => Def::NonMacroAttr(attr_kind),
            Def::Err => Def::Err,
        }
    }
}
