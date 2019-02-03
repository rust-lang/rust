use hir::def_id::DefId;
use util::nodemap::{NodeMap, DefIdMap};
use syntax::ast;
use syntax::ext::base::MacroKind;
use syntax_pos::Span;
use hir;
use ty;

use self::Namespace::*;

#[derive(Clone, Copy, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum CtorKind {
    /// Constructor function automatically created by a tuple struct/variant.
    Fn,
    /// Constructor constant automatically created by a unit struct/variant.
    Const,
    /// Unusable name in value namespace created by a struct variant.
    Fictive,
}

#[derive(Clone, Copy, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
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

#[derive(Clone, Copy, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum Def {
    // Type namespace
    Mod(DefId),
    Struct(DefId), // `DefId` refers to `NodeId` of the struct itself
    Union(DefId),
    Enum(DefId),
    Variant(DefId),
    Trait(DefId),
    /// `existential type Foo: Bar;`
    Existential(DefId),
    /// `type Foo = Bar;`
    TyAlias(DefId),
    ForeignTy(DefId),
    TraitAlias(DefId),
    AssociatedTy(DefId),
    /// `existential type Foo: Bar;`
    AssociatedExistential(DefId),
    PrimTy(hir::PrimTy),
    TyParam(DefId),
    SelfTy(Option<DefId> /* trait */, Option<DefId> /* impl */),
    ToolMod, // e.g., `rustfmt` in `#[rustfmt::skip]`

    // Value namespace
    Fn(DefId),
    Const(DefId),
    Static(DefId, bool /* is_mutbl */),
    StructCtor(DefId, CtorKind), // `DefId` refers to `NodeId` of the struct's constructor
    VariantCtor(DefId, CtorKind), // `DefId` refers to the enum variant
    SelfCtor(DefId /* impl */),  // `DefId` refers to the impl
    Method(DefId),
    AssociatedConst(DefId),

    Local(ast::NodeId),
    Upvar(ast::NodeId,  // `NodeId` of closed over local
          usize,        // index in the `freevars` list of the closure
          ast::NodeId), // expr node that creates the closure
    Label(ast::NodeId),

    // Macro namespace
    Macro(DefId, MacroKind),
    NonMacroAttr(NonMacroAttrKind), // e.g., `#[inline]` or `#[rustfmt::skip]`

    // Both namespaces
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
    base_def: Def,
    unresolved_segments: usize,
}

impl PathResolution {
    pub fn new(def: Def) -> Self {
        PathResolution { base_def: def, unresolved_segments: 0 }
    }

    pub fn with_unresolved_segments(def: Def, mut unresolved_segments: usize) -> Self {
        if def == Def::Err { unresolved_segments = 0 }
        PathResolution { base_def: def, unresolved_segments: unresolved_segments }
    }

    #[inline]
    pub fn base_def(&self) -> Def {
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
    /// Returns whether all the items in this collection are `None`.
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
pub type ExportMap = DefIdMap<Vec<Export>>;

/// Map used to track the `use` statements within a scope, matching it with all the items in every
/// namespace.
pub type ImportMap = NodeMap<PerNS<Option<PathResolution>>>;

#[derive(Copy, Clone, Debug, RustcEncodable, RustcDecodable)]
pub struct Export {
    /// The name of the target.
    pub ident: ast::Ident,
    /// The definition of the target.
    pub def: Def,
    /// The span of the target definition.
    pub span: Span,
    /// The visibility of the export.
    /// We include non-`pub` exports for hygienic macros that get used from extern crates.
    pub vis: ty::Visibility,
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

impl Def {
    /// Return the `DefId` of this `Def` if it has an id, else panic.
    pub fn def_id(&self) -> DefId {
        self.opt_def_id().unwrap_or_else(|| {
            bug!("attempted .def_id() on invalid def: {:?}", self)
        })
    }

    /// Return `Some(..)` with the `DefId` of this `Def` if it has a id, else `None`.
    pub fn opt_def_id(&self) -> Option<DefId> {
        match *self {
            Def::Fn(id) | Def::Mod(id) | Def::Static(id, _) |
            Def::Variant(id) | Def::VariantCtor(id, ..) | Def::Enum(id) |
            Def::TyAlias(id) | Def::TraitAlias(id) |
            Def::AssociatedTy(id) | Def::TyParam(id) | Def::Struct(id) | Def::StructCtor(id, ..) |
            Def::Union(id) | Def::Trait(id) | Def::Method(id) | Def::Const(id) |
            Def::AssociatedConst(id) | Def::Macro(id, ..) |
            Def::Existential(id) | Def::AssociatedExistential(id) | Def::ForeignTy(id) => {
                Some(id)
            }

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
            Def::Mod(id) => Some(id),
            _ => None,
        }
    }

    /// A human readable name for the def kind ("function", "module", etc.).
    pub fn kind_name(&self) -> &'static str {
        match *self {
            Def::Fn(..) => "function",
            Def::Mod(..) => "module",
            Def::Static(..) => "static",
            Def::Variant(..) => "variant",
            Def::VariantCtor(.., CtorKind::Fn) => "tuple variant",
            Def::VariantCtor(.., CtorKind::Const) => "unit variant",
            Def::VariantCtor(.., CtorKind::Fictive) => "struct variant",
            Def::Enum(..) => "enum",
            Def::Existential(..) => "existential type",
            Def::TyAlias(..) => "type alias",
            Def::TraitAlias(..) => "trait alias",
            Def::AssociatedTy(..) => "associated type",
            Def::AssociatedExistential(..) => "associated existential type",
            Def::Struct(..) => "struct",
            Def::StructCtor(.., CtorKind::Fn) => "tuple struct",
            Def::StructCtor(.., CtorKind::Const) => "unit struct",
            Def::StructCtor(.., CtorKind::Fictive) => bug!("impossible struct constructor"),
            Def::SelfCtor(..) => "self constructor",
            Def::Union(..) => "union",
            Def::Trait(..) => "trait",
            Def::ForeignTy(..) => "foreign type",
            Def::Method(..) => "method",
            Def::Const(..) => "constant",
            Def::AssociatedConst(..) => "associated constant",
            Def::TyParam(..) => "type parameter",
            Def::PrimTy(..) => "builtin type",
            Def::Local(..) => "local variable",
            Def::Upvar(..) => "closure capture",
            Def::Label(..) => "label",
            Def::SelfTy(..) => "self type",
            Def::Macro(.., macro_kind) => macro_kind.descr(),
            Def::ToolMod => "tool module",
            Def::NonMacroAttr(attr_kind) => attr_kind.descr(),
            Def::Err => "unresolved item",
        }
    }

    /// An English article for the def.
    pub fn article(&self) -> &'static str {
        match *self {
            Def::AssociatedTy(..) | Def::AssociatedConst(..) | Def::AssociatedExistential(..) |
            Def::Enum(..) | Def::Existential(..) | Def::Err => "an",
            Def::Macro(.., macro_kind) => macro_kind.article(),
            _ => "a",
        }
    }
}
