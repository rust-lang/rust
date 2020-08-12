//! A simplified AST that only contains items.

mod lower;
#[cfg(test)]
mod tests;

use std::{
    any::type_name,
    fmt::{self, Debug},
    hash::{Hash, Hasher},
    marker::PhantomData,
    ops::{Index, Range},
    sync::Arc,
};

use arena::{Arena, Idx, RawId};
use ast::{AstNode, AttrsOwner, NameOwner, StructKind};
use either::Either;
use hir_expand::{
    ast_id_map::FileAstId,
    hygiene::Hygiene,
    name::{name, AsName, Name},
    HirFileId, InFile,
};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use syntax::{ast, match_ast};
use test_utils::mark;

use crate::{
    attr::Attrs,
    db::DefDatabase,
    generics::GenericParams,
    path::{path, AssociatedTypeBinding, GenericArgs, ImportAlias, ModPath, Path, PathKind},
    type_ref::{Mutability, TypeBound, TypeRef},
    visibility::RawVisibility,
};

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct RawVisibilityId(u32);

impl RawVisibilityId {
    pub const PUB: Self = RawVisibilityId(u32::max_value());
    pub const PRIV: Self = RawVisibilityId(u32::max_value() - 1);
    pub const PUB_CRATE: Self = RawVisibilityId(u32::max_value() - 2);
}

impl fmt::Debug for RawVisibilityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut f = f.debug_tuple("RawVisibilityId");
        match *self {
            Self::PUB => f.field(&"pub"),
            Self::PRIV => f.field(&"pub(self)"),
            Self::PUB_CRATE => f.field(&"pub(crate)"),
            _ => f.field(&self.0),
        };
        f.finish()
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct GenericParamsId(u32);

impl GenericParamsId {
    pub const EMPTY: Self = GenericParamsId(u32::max_value());
}

/// The item tree of a source file.
#[derive(Debug, Eq, PartialEq)]
pub struct ItemTree {
    top_level: SmallVec<[ModItem; 1]>,
    attrs: FxHashMap<AttrOwner, Attrs>,
    inner_items: FxHashMap<FileAstId<ast::Item>, SmallVec<[ModItem; 1]>>,

    data: Option<Box<ItemTreeData>>,
}

impl ItemTree {
    pub fn item_tree_query(db: &dyn DefDatabase, file_id: HirFileId) -> Arc<ItemTree> {
        let _p = profile::span("item_tree_query").detail(|| format!("{:?}", file_id));
        let syntax = if let Some(node) = db.parse_or_expand(file_id) {
            node
        } else {
            return Arc::new(Self::empty());
        };

        let hygiene = Hygiene::new(db.upcast(), file_id);
        let ctx = lower::Ctx::new(db, hygiene.clone(), file_id);
        let mut top_attrs = None;
        let mut item_tree = match_ast! {
            match syntax {
                ast::SourceFile(file) => {
                    top_attrs = Some(Attrs::new(&file, &hygiene));
                    ctx.lower_module_items(&file)
                },
                ast::MacroItems(items) => {
                    ctx.lower_module_items(&items)
                },
                // Macros can expand to expressions. We return an empty item tree in this case, but
                // still need to collect inner items.
                ast::Expr(e) => {
                    ctx.lower_inner_items(e.syntax())
                },
                _ => {
                    panic!("cannot create item tree from {:?}", syntax);
                },
            }
        };

        if let Some(attrs) = top_attrs {
            item_tree.attrs.insert(AttrOwner::TopLevel, attrs);
        }
        item_tree.shrink_to_fit();
        Arc::new(item_tree)
    }

    fn empty() -> Self {
        Self {
            top_level: Default::default(),
            attrs: Default::default(),
            inner_items: Default::default(),
            data: Default::default(),
        }
    }

    fn shrink_to_fit(&mut self) {
        if let Some(data) = &mut self.data {
            let ItemTreeData {
                imports,
                extern_crates,
                functions,
                structs,
                fields,
                unions,
                enums,
                variants,
                consts,
                statics,
                traits,
                impls,
                type_aliases,
                mods,
                macro_calls,
                exprs,
                vis,
                generics,
            } = &mut **data;

            imports.shrink_to_fit();
            extern_crates.shrink_to_fit();
            functions.shrink_to_fit();
            structs.shrink_to_fit();
            fields.shrink_to_fit();
            unions.shrink_to_fit();
            enums.shrink_to_fit();
            variants.shrink_to_fit();
            consts.shrink_to_fit();
            statics.shrink_to_fit();
            traits.shrink_to_fit();
            impls.shrink_to_fit();
            type_aliases.shrink_to_fit();
            mods.shrink_to_fit();
            macro_calls.shrink_to_fit();
            exprs.shrink_to_fit();

            vis.arena.shrink_to_fit();
            generics.arena.shrink_to_fit();
        }
    }

    /// Returns an iterator over all items located at the top level of the `HirFileId` this
    /// `ItemTree` was created from.
    pub fn top_level_items(&self) -> &[ModItem] {
        &self.top_level
    }

    /// Returns the inner attributes of the source file.
    pub fn top_level_attrs(&self) -> &Attrs {
        self.attrs.get(&AttrOwner::TopLevel).unwrap_or(&Attrs::EMPTY)
    }

    pub fn attrs(&self, of: AttrOwner) -> &Attrs {
        self.attrs.get(&of).unwrap_or(&Attrs::EMPTY)
    }

    /// Returns the lowered inner items that `ast` corresponds to.
    ///
    /// Most AST items are lowered to a single `ModItem`, but some (eg. `use` items) may be lowered
    /// to multiple items in the `ItemTree`.
    pub fn inner_items(&self, ast: FileAstId<ast::Item>) -> &[ModItem] {
        &self.inner_items[&ast]
    }

    pub fn all_inner_items(&self) -> impl Iterator<Item = ModItem> + '_ {
        self.inner_items.values().flatten().copied()
    }

    pub fn source<S: ItemTreeNode>(&self, db: &dyn DefDatabase, of: ItemTreeId<S>) -> S::Source {
        // This unwrap cannot fail, since it has either succeeded above, or resulted in an empty
        // ItemTree (in which case there is no valid `FileItemTreeId` to call this method with).
        let root =
            db.parse_or_expand(of.file_id).expect("parse_or_expand failed on constructed ItemTree");

        let id = self[of.value].ast_id();
        let map = db.ast_id_map(of.file_id);
        let ptr = map.get(id);
        ptr.to_node(&root)
    }

    fn data(&self) -> &ItemTreeData {
        self.data.as_ref().expect("attempted to access data of empty ItemTree")
    }

    fn data_mut(&mut self) -> &mut ItemTreeData {
        self.data.get_or_insert_with(Box::default)
    }
}

#[derive(Default, Debug, Eq, PartialEq)]
struct ItemVisibilities {
    arena: Arena<RawVisibility>,
}

impl ItemVisibilities {
    fn alloc(&mut self, vis: RawVisibility) -> RawVisibilityId {
        match &vis {
            RawVisibility::Public => RawVisibilityId::PUB,
            RawVisibility::Module(path) if path.segments.is_empty() => match &path.kind {
                PathKind::Super(0) => RawVisibilityId::PRIV,
                PathKind::Crate => RawVisibilityId::PUB_CRATE,
                _ => RawVisibilityId(self.arena.alloc(vis).into_raw().into()),
            },
            _ => RawVisibilityId(self.arena.alloc(vis).into_raw().into()),
        }
    }
}

static VIS_PUB: RawVisibility = RawVisibility::Public;
static VIS_PRIV: RawVisibility =
    RawVisibility::Module(ModPath { kind: PathKind::Super(0), segments: Vec::new() });
static VIS_PUB_CRATE: RawVisibility =
    RawVisibility::Module(ModPath { kind: PathKind::Crate, segments: Vec::new() });

#[derive(Default, Debug, Eq, PartialEq)]
struct GenericParamsStorage {
    arena: Arena<GenericParams>,
}

impl GenericParamsStorage {
    fn alloc(&mut self, params: GenericParams) -> GenericParamsId {
        if params.types.is_empty() && params.where_predicates.is_empty() {
            return GenericParamsId::EMPTY;
        }

        GenericParamsId(self.arena.alloc(params).into_raw().into())
    }
}

static EMPTY_GENERICS: GenericParams =
    GenericParams { types: Arena::new(), where_predicates: Vec::new() };

#[derive(Default, Debug, Eq, PartialEq)]
struct ItemTreeData {
    imports: Arena<Import>,
    extern_crates: Arena<ExternCrate>,
    functions: Arena<Function>,
    structs: Arena<Struct>,
    fields: Arena<Field>,
    unions: Arena<Union>,
    enums: Arena<Enum>,
    variants: Arena<Variant>,
    consts: Arena<Const>,
    statics: Arena<Static>,
    traits: Arena<Trait>,
    impls: Arena<Impl>,
    type_aliases: Arena<TypeAlias>,
    mods: Arena<Mod>,
    macro_calls: Arena<MacroCall>,
    exprs: Arena<Expr>,

    vis: ItemVisibilities,
    generics: GenericParamsStorage,
}

#[derive(Debug, Eq, PartialEq, Hash)]
pub enum AttrOwner {
    /// Attributes on an item.
    ModItem(ModItem),
    /// Inner attributes of the source file.
    TopLevel,

    Variant(Idx<Variant>),
    Field(Idx<Field>),
    // FIXME: Store variant and field attrs, and stop reparsing them in `attrs_query`.
}

macro_rules! from_attrs {
    ( $( $var:ident($t:ty) ),+ ) => {
        $(
            impl From<$t> for AttrOwner {
                fn from(t: $t) -> AttrOwner {
                    AttrOwner::$var(t)
                }
            }
        )+
    };
}

from_attrs!(ModItem(ModItem), Variant(Idx<Variant>), Field(Idx<Field>));

/// Trait implemented by all item nodes in the item tree.
pub trait ItemTreeNode: Clone {
    type Source: AstNode + Into<ast::Item>;

    fn ast_id(&self) -> FileAstId<Self::Source>;

    /// Looks up an instance of `Self` in an item tree.
    fn lookup(tree: &ItemTree, index: Idx<Self>) -> &Self;

    /// Downcasts a `ModItem` to a `FileItemTreeId` specific to this type.
    fn id_from_mod_item(mod_item: ModItem) -> Option<FileItemTreeId<Self>>;

    /// Upcasts a `FileItemTreeId` to a generic `ModItem`.
    fn id_to_mod_item(id: FileItemTreeId<Self>) -> ModItem;
}

pub struct FileItemTreeId<N: ItemTreeNode> {
    index: Idx<N>,
    _p: PhantomData<N>,
}

impl<N: ItemTreeNode> Clone for FileItemTreeId<N> {
    fn clone(&self) -> Self {
        Self { index: self.index, _p: PhantomData }
    }
}
impl<N: ItemTreeNode> Copy for FileItemTreeId<N> {}

impl<N: ItemTreeNode> PartialEq for FileItemTreeId<N> {
    fn eq(&self, other: &FileItemTreeId<N>) -> bool {
        self.index == other.index
    }
}
impl<N: ItemTreeNode> Eq for FileItemTreeId<N> {}

impl<N: ItemTreeNode> Hash for FileItemTreeId<N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.index.hash(state)
    }
}

impl<N: ItemTreeNode> fmt::Debug for FileItemTreeId<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.index.fmt(f)
    }
}

pub type ItemTreeId<N> = InFile<FileItemTreeId<N>>;

macro_rules! mod_items {
    ( $( $typ:ident in $fld:ident -> $ast:ty ),+ $(,)? ) => {
        #[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
        pub enum ModItem {
            $(
                $typ(FileItemTreeId<$typ>),
            )+
        }

        $(
            impl From<FileItemTreeId<$typ>> for ModItem {
                fn from(id: FileItemTreeId<$typ>) -> ModItem {
                    ModItem::$typ(id)
                }
            }
        )+

        $(
            impl ItemTreeNode for $typ {
                type Source = $ast;

                fn ast_id(&self) -> FileAstId<Self::Source> {
                    self.ast_id
                }

                fn lookup(tree: &ItemTree, index: Idx<Self>) -> &Self {
                    &tree.data().$fld[index]
                }

                fn id_from_mod_item(mod_item: ModItem) -> Option<FileItemTreeId<Self>> {
                    if let ModItem::$typ(id) = mod_item {
                        Some(id)
                    } else {
                        None
                    }
                }

                fn id_to_mod_item(id: FileItemTreeId<Self>) -> ModItem {
                    ModItem::$typ(id)
                }
            }

            impl Index<Idx<$typ>> for ItemTree {
                type Output = $typ;

                fn index(&self, index: Idx<$typ>) -> &Self::Output {
                    &self.data().$fld[index]
                }
            }
        )+
    };
}

mod_items! {
    Import in imports -> ast::Use,
    ExternCrate in extern_crates -> ast::ExternCrate,
    Function in functions -> ast::Fn,
    Struct in structs -> ast::Struct,
    Union in unions -> ast::Union,
    Enum in enums -> ast::Enum,
    Const in consts -> ast::Const,
    Static in statics -> ast::Static,
    Trait in traits -> ast::Trait,
    Impl in impls -> ast::Impl,
    TypeAlias in type_aliases -> ast::TypeAlias,
    Mod in mods -> ast::Module,
    MacroCall in macro_calls -> ast::MacroCall,
}

macro_rules! impl_index {
    ( $($fld:ident: $t:ty),+ $(,)? ) => {
        $(
            impl Index<Idx<$t>> for ItemTree {
                type Output = $t;

                fn index(&self, index: Idx<$t>) -> &Self::Output {
                    &self.data().$fld[index]
                }
            }
        )+
    };
}

impl_index!(fields: Field, variants: Variant, exprs: Expr);

impl Index<RawVisibilityId> for ItemTree {
    type Output = RawVisibility;
    fn index(&self, index: RawVisibilityId) -> &Self::Output {
        match index {
            RawVisibilityId::PRIV => &VIS_PRIV,
            RawVisibilityId::PUB => &VIS_PUB,
            RawVisibilityId::PUB_CRATE => &VIS_PUB_CRATE,
            _ => &self.data().vis.arena[Idx::from_raw(index.0.into())],
        }
    }
}

impl Index<GenericParamsId> for ItemTree {
    type Output = GenericParams;

    fn index(&self, index: GenericParamsId) -> &Self::Output {
        match index {
            GenericParamsId::EMPTY => &EMPTY_GENERICS,
            _ => &self.data().generics.arena[Idx::from_raw(index.0.into())],
        }
    }
}

impl<N: ItemTreeNode> Index<FileItemTreeId<N>> for ItemTree {
    type Output = N;
    fn index(&self, id: FileItemTreeId<N>) -> &N {
        N::lookup(self, id.index)
    }
}

/// A desugared `use` import.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Import {
    pub path: ModPath,
    pub alias: Option<ImportAlias>,
    pub visibility: RawVisibilityId,
    pub is_glob: bool,
    pub is_prelude: bool,
    /// AST ID of the `use` or `extern crate` item this import was derived from. Note that many
    /// `Import`s can map to the same `use` item.
    pub ast_id: FileAstId<ast::Use>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ExternCrate {
    pub path: ModPath,
    pub alias: Option<ImportAlias>,
    pub visibility: RawVisibilityId,
    /// Whether this is a `#[macro_use] extern crate ...`.
    pub is_macro_use: bool,
    pub ast_id: FileAstId<ast::ExternCrate>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Function {
    pub name: Name,
    pub visibility: RawVisibilityId,
    pub generic_params: GenericParamsId,
    pub has_self_param: bool,
    pub is_unsafe: bool,
    pub params: Box<[TypeRef]>,
    pub is_varargs: bool,
    pub ret_type: TypeRef,
    pub ast_id: FileAstId<ast::Fn>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Struct {
    pub name: Name,
    pub visibility: RawVisibilityId,
    pub generic_params: GenericParamsId,
    pub fields: Fields,
    pub ast_id: FileAstId<ast::Struct>,
    pub kind: StructDefKind,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum StructDefKind {
    /// `struct S { ... }` - type namespace only.
    Record,
    /// `struct S(...);`
    Tuple,
    /// `struct S;`
    Unit,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Union {
    pub name: Name,
    pub visibility: RawVisibilityId,
    pub generic_params: GenericParamsId,
    pub fields: Fields,
    pub ast_id: FileAstId<ast::Union>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Enum {
    pub name: Name,
    pub visibility: RawVisibilityId,
    pub generic_params: GenericParamsId,
    pub variants: IdRange<Variant>,
    pub ast_id: FileAstId<ast::Enum>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Const {
    /// const _: () = ();
    pub name: Option<Name>,
    pub visibility: RawVisibilityId,
    pub type_ref: TypeRef,
    pub ast_id: FileAstId<ast::Const>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Static {
    pub name: Name,
    pub visibility: RawVisibilityId,
    pub mutable: bool,
    pub type_ref: TypeRef,
    pub ast_id: FileAstId<ast::Static>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Trait {
    pub name: Name,
    pub visibility: RawVisibilityId,
    pub generic_params: GenericParamsId,
    pub auto: bool,
    pub items: Box<[AssocItem]>,
    pub ast_id: FileAstId<ast::Trait>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Impl {
    pub generic_params: GenericParamsId,
    pub target_trait: Option<TypeRef>,
    pub target_type: TypeRef,
    pub is_negative: bool,
    pub items: Box<[AssocItem]>,
    pub ast_id: FileAstId<ast::Impl>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeAlias {
    pub name: Name,
    pub visibility: RawVisibilityId,
    /// Bounds on the type alias itself. Only valid in trait declarations, eg. `type Assoc: Copy;`.
    pub bounds: Box<[TypeBound]>,
    pub generic_params: GenericParamsId,
    pub type_ref: Option<TypeRef>,
    pub ast_id: FileAstId<ast::TypeAlias>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Mod {
    pub name: Name,
    pub visibility: RawVisibilityId,
    pub kind: ModKind,
    pub ast_id: FileAstId<ast::Module>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ModKind {
    /// `mod m { ... }`
    Inline { items: Box<[ModItem]> },

    /// `mod m;`
    Outline {},
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MacroCall {
    /// For `macro_rules!` declarations, this is the name of the declared macro.
    pub name: Option<Name>,
    /// Path to the called macro.
    pub path: ModPath,
    /// Has `#[macro_export]`.
    pub is_export: bool,
    /// Has `#[macro_export(local_inner_macros)]`.
    pub is_local_inner: bool,
    /// Has `#[rustc_builtin_macro]`.
    pub is_builtin: bool,
    pub ast_id: FileAstId<ast::MacroCall>,
}

// NB: There's no `FileAstId` for `Expr`. The only case where this would be useful is for array
// lengths, but we don't do much with them yet.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Expr;

macro_rules! impl_froms {
    ($e:ident { $($v:ident ($t:ty)),* $(,)? }) => {
        $(
            impl From<$t> for $e {
                fn from(it: $t) -> $e {
                    $e::$v(it)
                }
            }
        )*
    }
}

impl ModItem {
    pub fn as_assoc_item(&self) -> Option<AssocItem> {
        match self {
            ModItem::Import(_)
            | ModItem::ExternCrate(_)
            | ModItem::Struct(_)
            | ModItem::Union(_)
            | ModItem::Enum(_)
            | ModItem::Static(_)
            | ModItem::Trait(_)
            | ModItem::Impl(_)
            | ModItem::Mod(_) => None,
            ModItem::MacroCall(call) => Some(AssocItem::MacroCall(*call)),
            ModItem::Const(konst) => Some(AssocItem::Const(*konst)),
            ModItem::TypeAlias(alias) => Some(AssocItem::TypeAlias(*alias)),
            ModItem::Function(func) => Some(AssocItem::Function(*func)),
        }
    }

    pub fn downcast<N: ItemTreeNode>(self) -> Option<FileItemTreeId<N>> {
        N::id_from_mod_item(self)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum AssocItem {
    Function(FileItemTreeId<Function>),
    TypeAlias(FileItemTreeId<TypeAlias>),
    Const(FileItemTreeId<Const>),
    MacroCall(FileItemTreeId<MacroCall>),
}

impl_froms!(AssocItem {
    Function(FileItemTreeId<Function>),
    TypeAlias(FileItemTreeId<TypeAlias>),
    Const(FileItemTreeId<Const>),
    MacroCall(FileItemTreeId<MacroCall>),
});

impl From<AssocItem> for ModItem {
    fn from(item: AssocItem) -> Self {
        match item {
            AssocItem::Function(it) => it.into(),
            AssocItem::TypeAlias(it) => it.into(),
            AssocItem::Const(it) => it.into(),
            AssocItem::MacroCall(it) => it.into(),
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct Variant {
    pub name: Name,
    pub fields: Fields,
}

pub struct IdRange<T> {
    range: Range<u32>,
    _p: PhantomData<T>,
}

impl<T> IdRange<T> {
    fn new(range: Range<Idx<T>>) -> Self {
        Self { range: range.start.into_raw().into()..range.end.into_raw().into(), _p: PhantomData }
    }
}

impl<T> Iterator for IdRange<T> {
    type Item = Idx<T>;
    fn next(&mut self) -> Option<Self::Item> {
        self.range.next().map(|raw| Idx::from_raw(raw.into()))
    }
}

impl<T> fmt::Debug for IdRange<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple(&format!("IdRange::<{}>", type_name::<T>())).field(&self.range).finish()
    }
}

impl<T> Clone for IdRange<T> {
    fn clone(&self) -> Self {
        Self { range: self.range.clone(), _p: PhantomData }
    }
}

impl<T> PartialEq for IdRange<T> {
    fn eq(&self, other: &Self) -> bool {
        self.range == other.range
    }
}

impl<T> Eq for IdRange<T> {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fields {
    Record(IdRange<Field>),
    Tuple(IdRange<Field>),
    Unit,
}

/// A single field of an enum variant or struct
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Field {
    pub name: Name,
    pub type_ref: TypeRef,
    pub visibility: RawVisibilityId,
}
