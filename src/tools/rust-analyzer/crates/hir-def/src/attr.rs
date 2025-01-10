//! A higher level attributes based on TokenTree, with also some shortcuts.

use std::{borrow::Cow, hash::Hash, ops};

use base_db::CrateId;
use cfg::{CfgExpr, CfgOptions};
use either::Either;
use hir_expand::{
    attrs::{collect_attrs, Attr, AttrId, RawAttrs},
    HirFileId, InFile,
};
use intern::{sym, Symbol};
use la_arena::{ArenaMap, Idx, RawIdx};
use mbe::DelimiterKind;
use syntax::{
    ast::{self, HasAttrs},
    AstPtr,
};
use triomphe::Arc;
use tt::iter::{TtElement, TtIter};

use crate::{
    db::DefDatabase,
    item_tree::{AttrOwner, FieldParent, ItemTreeNode},
    lang_item::LangItem,
    nameres::{ModuleOrigin, ModuleSource},
    src::{HasChildSource, HasSource},
    AdtId, AttrDefId, GenericParamId, HasModule, ItemTreeLoc, LocalFieldId, Lookup, MacroId,
    VariantId,
};

/// Desugared attributes of an item post `cfg_attr` expansion.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct Attrs(RawAttrs);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttrsWithOwner {
    attrs: Attrs,
    owner: AttrDefId,
}

impl Attrs {
    pub fn get(&self, id: AttrId) -> Option<&Attr> {
        (**self).iter().find(|attr| attr.id == id)
    }

    pub(crate) fn filter(db: &dyn DefDatabase, krate: CrateId, raw_attrs: RawAttrs) -> Attrs {
        Attrs(raw_attrs.filter(db.upcast(), krate))
    }
}

impl ops::Deref for Attrs {
    type Target = [Attr];

    fn deref(&self) -> &[Attr] {
        &self.0
    }
}

impl ops::Deref for AttrsWithOwner {
    type Target = Attrs;

    fn deref(&self) -> &Attrs {
        &self.attrs
    }
}

impl Attrs {
    pub const EMPTY: Self = Self(RawAttrs::EMPTY);

    pub(crate) fn fields_attrs_query(
        db: &dyn DefDatabase,
        v: VariantId,
    ) -> Arc<ArenaMap<LocalFieldId, Attrs>> {
        let _p = tracing::info_span!("fields_attrs_query").entered();
        // FIXME: There should be some proper form of mapping between item tree field ids and hir field ids
        let mut res = ArenaMap::default();

        let crate_graph = db.crate_graph();
        let item_tree;
        let (parent, fields, krate) = match v {
            VariantId::EnumVariantId(it) => {
                let loc = it.lookup(db);
                let krate = loc.parent.lookup(db).container.krate;
                item_tree = loc.id.item_tree(db);
                let variant = &item_tree[loc.id.value];
                (FieldParent::Variant(loc.id.value), &variant.fields, krate)
            }
            VariantId::StructId(it) => {
                let loc = it.lookup(db);
                let krate = loc.container.krate;
                item_tree = loc.id.item_tree(db);
                let struct_ = &item_tree[loc.id.value];
                (FieldParent::Struct(loc.id.value), &struct_.fields, krate)
            }
            VariantId::UnionId(it) => {
                let loc = it.lookup(db);
                let krate = loc.container.krate;
                item_tree = loc.id.item_tree(db);
                let union_ = &item_tree[loc.id.value];
                (FieldParent::Union(loc.id.value), &union_.fields, krate)
            }
        };

        let cfg_options = &crate_graph[krate].cfg_options;

        let mut idx = 0;
        for (id, _field) in fields.iter().enumerate() {
            let attrs = item_tree.attrs(db, krate, AttrOwner::make_field_indexed(parent, id));
            if attrs.is_cfg_enabled(cfg_options) {
                res.insert(Idx::from_raw(RawIdx::from(idx)), attrs);
                idx += 1;
            }
        }

        Arc::new(res)
    }
}

impl Attrs {
    pub fn by_key<'attrs>(&'attrs self, key: &'attrs Symbol) -> AttrQuery<'attrs> {
        AttrQuery { attrs: self, key }
    }

    pub fn rust_analyzer_tool(&self) -> impl Iterator<Item = &Attr> {
        self.iter()
            .filter(|&attr| attr.path.segments().first().is_some_and(|s| *s == sym::rust_analyzer))
    }

    pub fn cfg(&self) -> Option<CfgExpr> {
        let mut cfgs = self.by_key(&sym::cfg).tt_values().map(CfgExpr::parse);
        let first = cfgs.next()?;
        match cfgs.next() {
            Some(second) => {
                let cfgs = [first, second].into_iter().chain(cfgs);
                Some(CfgExpr::All(cfgs.collect()))
            }
            None => Some(first),
        }
    }

    pub fn cfgs(&self) -> impl Iterator<Item = CfgExpr> + '_ {
        self.by_key(&sym::cfg).tt_values().map(CfgExpr::parse)
    }

    pub(crate) fn is_cfg_enabled(&self, cfg_options: &CfgOptions) -> bool {
        match self.cfg() {
            None => true,
            Some(cfg) => cfg_options.check(&cfg) != Some(false),
        }
    }

    pub fn lang(&self) -> Option<&Symbol> {
        self.by_key(&sym::lang).string_value()
    }

    pub fn lang_item(&self) -> Option<LangItem> {
        self.by_key(&sym::lang).string_value().and_then(LangItem::from_symbol)
    }

    pub fn has_doc_hidden(&self) -> bool {
        self.by_key(&sym::doc).tt_values().any(|tt| {
            tt.top_subtree().delimiter.kind == DelimiterKind::Parenthesis &&
                matches!(tt.token_trees().flat_tokens(), [tt::TokenTree::Leaf(tt::Leaf::Ident(ident))] if ident.sym == sym::hidden)
        })
    }

    pub fn has_doc_notable_trait(&self) -> bool {
        self.by_key(&sym::doc).tt_values().any(|tt| {
            tt.top_subtree().delimiter.kind == DelimiterKind::Parenthesis &&
                matches!(tt.token_trees().flat_tokens(), [tt::TokenTree::Leaf(tt::Leaf::Ident(ident))] if ident.sym == sym::notable_trait)
        })
    }

    pub fn doc_exprs(&self) -> impl Iterator<Item = DocExpr> + '_ {
        self.by_key(&sym::doc).tt_values().map(DocExpr::parse)
    }

    pub fn doc_aliases(&self) -> impl Iterator<Item = Symbol> + '_ {
        self.doc_exprs().flat_map(|doc_expr| doc_expr.aliases().to_vec())
    }

    pub fn export_name(&self) -> Option<&Symbol> {
        self.by_key(&sym::export_name).string_value()
    }

    pub fn is_proc_macro(&self) -> bool {
        self.by_key(&sym::proc_macro).exists()
    }

    pub fn is_proc_macro_attribute(&self) -> bool {
        self.by_key(&sym::proc_macro_attribute).exists()
    }

    pub fn is_proc_macro_derive(&self) -> bool {
        self.by_key(&sym::proc_macro_derive).exists()
    }

    pub fn is_test(&self) -> bool {
        self.iter().any(|it| {
            it.path()
                .segments()
                .iter()
                .rev()
                .zip(
                    [sym::core.clone(), sym::prelude.clone(), sym::v1.clone(), sym::test.clone()]
                        .iter()
                        .rev(),
                )
                .all(|it| it.0 == it.1)
        })
    }

    pub fn is_ignore(&self) -> bool {
        self.by_key(&sym::ignore).exists()
    }

    pub fn is_bench(&self) -> bool {
        self.by_key(&sym::bench).exists()
    }

    pub fn is_unstable(&self) -> bool {
        self.by_key(&sym::unstable).exists()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DocAtom {
    /// eg. `#[doc(hidden)]`
    Flag(Symbol),
    /// eg. `#[doc(alias = "it")]`
    ///
    /// Note that a key can have multiple values that are all considered "active" at the same time.
    /// For example, `#[doc(alias = "x")]` and `#[doc(alias = "y")]`.
    KeyValue { key: Symbol, value: Symbol },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DocExpr {
    Invalid,
    /// eg. `#[doc(hidden)]`, `#[doc(alias = "x")]`
    Atom(DocAtom),
    /// eg. `#[doc(alias("x", "y"))]`
    Alias(Vec<Symbol>),
}

impl From<DocAtom> for DocExpr {
    fn from(atom: DocAtom) -> Self {
        DocExpr::Atom(atom)
    }
}

impl DocExpr {
    fn parse<S: Copy>(tt: &tt::TopSubtree<S>) -> DocExpr {
        next_doc_expr(tt.iter()).unwrap_or(DocExpr::Invalid)
    }

    pub fn aliases(&self) -> &[Symbol] {
        match self {
            DocExpr::Atom(DocAtom::KeyValue { key, value }) if *key == sym::alias => {
                std::slice::from_ref(value)
            }
            DocExpr::Alias(aliases) => aliases,
            _ => &[],
        }
    }
}

fn next_doc_expr<S: Copy>(mut it: TtIter<'_, S>) -> Option<DocExpr> {
    let name = match it.next() {
        None => return None,
        Some(TtElement::Leaf(tt::Leaf::Ident(ident))) => ident.sym.clone(),
        Some(_) => return Some(DocExpr::Invalid),
    };

    // Peek
    let ret = match it.peek() {
        Some(TtElement::Leaf(tt::Leaf::Punct(punct))) if punct.char == '=' => {
            it.next();
            match it.next() {
                Some(TtElement::Leaf(tt::Leaf::Literal(tt::Literal {
                    symbol: text,
                    kind: tt::LitKind::Str,
                    ..
                }))) => DocAtom::KeyValue { key: name, value: text.clone() }.into(),
                _ => return Some(DocExpr::Invalid),
            }
        }
        Some(TtElement::Subtree(_, subtree_iter)) => {
            it.next();
            let subs = parse_comma_sep(subtree_iter);
            match &name {
                s if *s == sym::alias => DocExpr::Alias(subs),
                _ => DocExpr::Invalid,
            }
        }
        _ => DocAtom::Flag(name).into(),
    };
    Some(ret)
}

fn parse_comma_sep<S>(iter: TtIter<'_, S>) -> Vec<Symbol> {
    iter.filter_map(|tt| match tt {
        TtElement::Leaf(tt::Leaf::Literal(tt::Literal {
            kind: tt::LitKind::Str, symbol, ..
        })) => Some(symbol.clone()),
        _ => None,
    })
    .collect()
}

impl AttrsWithOwner {
    pub fn new(db: &dyn DefDatabase, owner: AttrDefId) -> Self {
        Self { attrs: db.attrs(owner), owner }
    }

    pub(crate) fn attrs_query(db: &dyn DefDatabase, def: AttrDefId) -> Attrs {
        let _p = tracing::info_span!("attrs_query").entered();
        // FIXME: this should use `Trace` to avoid duplication in `source_map` below
        let raw_attrs = match def {
            AttrDefId::ModuleId(module) => {
                let def_map = module.def_map(db);
                let mod_data = &def_map[module.local_id];

                match mod_data.origin {
                    ModuleOrigin::File { definition, declaration_tree_id, .. } => {
                        let decl_attrs = declaration_tree_id
                            .item_tree(db)
                            .raw_attrs(AttrOwner::ModItem(declaration_tree_id.value.into()))
                            .clone();
                        let tree = db.file_item_tree(definition.into());
                        let def_attrs = tree.raw_attrs(AttrOwner::TopLevel).clone();
                        decl_attrs.merge(def_attrs)
                    }
                    ModuleOrigin::CrateRoot { definition } => {
                        let tree = db.file_item_tree(definition.into());
                        tree.raw_attrs(AttrOwner::TopLevel).clone()
                    }
                    ModuleOrigin::Inline { definition_tree_id, .. } => definition_tree_id
                        .item_tree(db)
                        .raw_attrs(AttrOwner::ModItem(definition_tree_id.value.into()))
                        .clone(),
                    ModuleOrigin::BlockExpr { id, .. } => {
                        let tree = db.block_item_tree(id);
                        tree.raw_attrs(AttrOwner::TopLevel).clone()
                    }
                }
            }
            AttrDefId::FieldId(it) => {
                return db.fields_attrs(it.parent)[it.local_id].clone();
            }
            AttrDefId::EnumVariantId(it) => attrs_from_item_tree_loc(db, it),
            AttrDefId::AdtId(it) => match it {
                AdtId::StructId(it) => attrs_from_item_tree_loc(db, it),
                AdtId::EnumId(it) => attrs_from_item_tree_loc(db, it),
                AdtId::UnionId(it) => attrs_from_item_tree_loc(db, it),
            },
            AttrDefId::TraitId(it) => attrs_from_item_tree_loc(db, it),
            AttrDefId::TraitAliasId(it) => attrs_from_item_tree_loc(db, it),
            AttrDefId::MacroId(it) => match it {
                MacroId::Macro2Id(it) => attrs_from_item_tree_loc(db, it),
                MacroId::MacroRulesId(it) => attrs_from_item_tree_loc(db, it),
                MacroId::ProcMacroId(it) => attrs_from_item_tree_loc(db, it),
            },
            AttrDefId::ImplId(it) => attrs_from_item_tree_loc(db, it),
            AttrDefId::ConstId(it) => attrs_from_item_tree_loc(db, it),
            AttrDefId::StaticId(it) => attrs_from_item_tree_loc(db, it),
            AttrDefId::FunctionId(it) => attrs_from_item_tree_loc(db, it),
            AttrDefId::TypeAliasId(it) => attrs_from_item_tree_loc(db, it),
            AttrDefId::GenericParamId(it) => match it {
                GenericParamId::ConstParamId(it) => {
                    let src = it.parent().child_source(db);
                    // FIXME: We should be never getting `None` here.
                    match src.value.get(it.local_id()) {
                        Some(val) => RawAttrs::from_attrs_owner(
                            db.upcast(),
                            src.with_value(val),
                            db.span_map(src.file_id).as_ref(),
                        ),
                        None => RawAttrs::EMPTY,
                    }
                }
                GenericParamId::TypeParamId(it) => {
                    let src = it.parent().child_source(db);
                    // FIXME: We should be never getting `None` here.
                    match src.value.get(it.local_id()) {
                        Some(val) => RawAttrs::from_attrs_owner(
                            db.upcast(),
                            src.with_value(val),
                            db.span_map(src.file_id).as_ref(),
                        ),
                        None => RawAttrs::EMPTY,
                    }
                }
                GenericParamId::LifetimeParamId(it) => {
                    let src = it.parent.child_source(db);
                    // FIXME: We should be never getting `None` here.
                    match src.value.get(it.local_id) {
                        Some(val) => RawAttrs::from_attrs_owner(
                            db.upcast(),
                            src.with_value(val),
                            db.span_map(src.file_id).as_ref(),
                        ),
                        None => RawAttrs::EMPTY,
                    }
                }
            },
            AttrDefId::ExternBlockId(it) => attrs_from_item_tree_loc(db, it),
            AttrDefId::ExternCrateId(it) => attrs_from_item_tree_loc(db, it),
            AttrDefId::UseId(it) => attrs_from_item_tree_loc(db, it),
        };

        let attrs = raw_attrs.filter(db.upcast(), def.krate(db));
        Attrs(attrs)
    }

    pub fn source_map(&self, db: &dyn DefDatabase) -> AttrSourceMap {
        let owner = match self.owner {
            AttrDefId::ModuleId(module) => {
                // Modules can have 2 attribute owners (the `mod x;` item, and the module file itself).

                let def_map = module.def_map(db);
                let mod_data = &def_map[module.local_id];
                match mod_data.declaration_source(db) {
                    Some(it) => {
                        let mut map = AttrSourceMap::new(InFile::new(it.file_id, &it.value));
                        if let InFile { file_id, value: ModuleSource::SourceFile(file) } =
                            mod_data.definition_source(db)
                        {
                            map.append_module_inline_attrs(AttrSourceMap::new(InFile::new(
                                file_id, &file,
                            )));
                        }
                        return map;
                    }
                    None => {
                        let InFile { file_id, value } = mod_data.definition_source(db);
                        let attrs_owner = match &value {
                            ModuleSource::SourceFile(file) => file as &dyn ast::HasAttrs,
                            ModuleSource::Module(module) => module as &dyn ast::HasAttrs,
                            ModuleSource::BlockExpr(block) => block as &dyn ast::HasAttrs,
                        };
                        return AttrSourceMap::new(InFile::new(file_id, attrs_owner));
                    }
                }
            }
            AttrDefId::FieldId(id) => {
                let map = db.fields_attrs_source_map(id.parent);
                let file_id = id.parent.file_id(db);
                let root = db.parse_or_expand(file_id);
                let owner = ast::AnyHasAttrs::new(map[id.local_id].to_node(&root));
                InFile::new(file_id, owner)
            }
            AttrDefId::AdtId(adt) => match adt {
                AdtId::StructId(id) => any_has_attrs(db, id),
                AdtId::UnionId(id) => any_has_attrs(db, id),
                AdtId::EnumId(id) => any_has_attrs(db, id),
            },
            AttrDefId::FunctionId(id) => any_has_attrs(db, id),
            AttrDefId::EnumVariantId(id) => any_has_attrs(db, id),
            AttrDefId::StaticId(id) => any_has_attrs(db, id),
            AttrDefId::ConstId(id) => any_has_attrs(db, id),
            AttrDefId::TraitId(id) => any_has_attrs(db, id),
            AttrDefId::TraitAliasId(id) => any_has_attrs(db, id),
            AttrDefId::TypeAliasId(id) => any_has_attrs(db, id),
            AttrDefId::MacroId(id) => match id {
                MacroId::Macro2Id(id) => any_has_attrs(db, id),
                MacroId::MacroRulesId(id) => any_has_attrs(db, id),
                MacroId::ProcMacroId(id) => any_has_attrs(db, id),
            },
            AttrDefId::ImplId(id) => any_has_attrs(db, id),
            AttrDefId::GenericParamId(id) => match id {
                GenericParamId::ConstParamId(id) => id
                    .parent()
                    .child_source(db)
                    .map(|source| ast::AnyHasAttrs::new(source[id.local_id()].clone())),
                GenericParamId::TypeParamId(id) => id
                    .parent()
                    .child_source(db)
                    .map(|source| ast::AnyHasAttrs::new(source[id.local_id()].clone())),
                GenericParamId::LifetimeParamId(id) => id
                    .parent
                    .child_source(db)
                    .map(|source| ast::AnyHasAttrs::new(source[id.local_id].clone())),
            },
            AttrDefId::ExternBlockId(id) => any_has_attrs(db, id),
            AttrDefId::ExternCrateId(id) => any_has_attrs(db, id),
            AttrDefId::UseId(id) => any_has_attrs(db, id),
        };

        AttrSourceMap::new(owner.as_ref().map(|node| node as &dyn HasAttrs))
    }
}

#[derive(Debug)]
pub struct AttrSourceMap {
    source: Vec<Either<ast::Attr, ast::Comment>>,
    file_id: HirFileId,
    /// If this map is for a module, this will be the [`HirFileId`] of the module's definition site,
    /// while `file_id` will be the one of the module declaration site.
    /// The usize is the index into `source` from which point on the entries reside in the def site
    /// file.
    mod_def_site_file_id: Option<(HirFileId, usize)>,
}

impl AttrSourceMap {
    fn new(owner: InFile<&dyn ast::HasAttrs>) -> Self {
        Self {
            source: collect_attrs(owner.value).map(|(_, it)| it).collect(),
            file_id: owner.file_id,
            mod_def_site_file_id: None,
        }
    }

    /// Append a second source map to this one, this is required for modules, whose outline and inline
    /// attributes can reside in different files
    fn append_module_inline_attrs(&mut self, other: Self) {
        assert!(self.mod_def_site_file_id.is_none() && other.mod_def_site_file_id.is_none());
        let len = self.source.len();
        self.source.extend(other.source);
        if other.file_id != self.file_id {
            self.mod_def_site_file_id = Some((other.file_id, len));
        }
    }

    /// Maps the lowered `Attr` back to its original syntax node.
    ///
    /// `attr` must come from the `owner` used for AttrSourceMap
    ///
    /// Note that the returned syntax node might be a `#[cfg_attr]`, or a doc comment, instead of
    /// the attribute represented by `Attr`.
    pub fn source_of(&self, attr: &Attr) -> InFile<&Either<ast::Attr, ast::Comment>> {
        self.source_of_id(attr.id)
    }

    pub fn source_of_id(&self, id: AttrId) -> InFile<&Either<ast::Attr, ast::Comment>> {
        let ast_idx = id.ast_index();
        let file_id = match self.mod_def_site_file_id {
            Some((file_id, def_site_cut)) if def_site_cut <= ast_idx => file_id,
            _ => self.file_id,
        };

        self.source
            .get(ast_idx)
            .map(|it| InFile::new(file_id, it))
            .unwrap_or_else(|| panic!("cannot find attr at index {id:?}"))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AttrQuery<'attr> {
    attrs: &'attr Attrs,
    key: &'attr Symbol,
}

impl<'attr> AttrQuery<'attr> {
    pub fn tt_values(self) -> impl Iterator<Item = &'attr crate::tt::TopSubtree> {
        self.attrs().filter_map(|attr| attr.token_tree_value())
    }

    pub fn string_value(self) -> Option<&'attr Symbol> {
        self.attrs().find_map(|attr| attr.string_value())
    }

    pub fn string_value_with_span(self) -> Option<(&'attr Symbol, span::Span)> {
        self.attrs().find_map(|attr| attr.string_value_with_span())
    }

    pub fn string_value_unescape(self) -> Option<Cow<'attr, str>> {
        self.attrs().find_map(|attr| attr.string_value_unescape())
    }

    pub fn exists(self) -> bool {
        self.attrs().next().is_some()
    }

    pub fn attrs(self) -> impl Iterator<Item = &'attr Attr> + Clone {
        let key = self.key;
        self.attrs.iter().filter(move |attr| attr.path.as_ident().is_some_and(|s| *s == *key))
    }

    /// Find string value for a specific key inside token tree
    ///
    /// ```ignore
    /// #[doc(html_root_url = "url")]
    ///       ^^^^^^^^^^^^^ key
    /// ```
    pub fn find_string_value_in_tt(self, key: &'attr Symbol) -> Option<&'attr str> {
        self.tt_values().find_map(|tt| {
            let name = tt.iter()
                .skip_while(|tt| !matches!(tt, TtElement::Leaf(tt::Leaf::Ident(tt::Ident { sym, ..} )) if *sym == *key))
                .nth(2);

            match name {
                Some(TtElement::Leaf(tt::Leaf::Literal(tt::Literal{  symbol: text, kind: tt::LitKind::Str | tt::LitKind::StrRaw(_) , ..}))) => Some(text.as_str()),
                _ => None
            }
        })
    }
}

fn any_has_attrs<'db>(
    db: &(dyn DefDatabase + 'db),
    id: impl Lookup<
        Database<'db> = dyn DefDatabase + 'db,
        Data = impl HasSource<Value = impl ast::HasAttrs>,
    >,
) -> InFile<ast::AnyHasAttrs> {
    id.lookup(db).source(db).map(ast::AnyHasAttrs::new)
}

fn attrs_from_item_tree_loc<'db, N: ItemTreeNode>(
    db: &(dyn DefDatabase + 'db),
    lookup: impl Lookup<Database<'db> = dyn DefDatabase + 'db, Data = impl ItemTreeLoc<Id = N>>,
) -> RawAttrs {
    let id = lookup.lookup(db).item_tree_id();
    let tree = id.item_tree(db);
    let attr_owner = N::attr_owner(id.value);
    tree.raw_attrs(attr_owner).clone()
}

pub(crate) fn fields_attrs_source_map(
    db: &dyn DefDatabase,
    def: VariantId,
) -> Arc<ArenaMap<LocalFieldId, AstPtr<Either<ast::TupleField, ast::RecordField>>>> {
    let mut res = ArenaMap::default();
    let child_source = def.child_source(db);

    for (idx, variant) in child_source.value.iter() {
        res.insert(
            idx,
            variant
                .as_ref()
                .either(|l| AstPtr::new(l).wrap_left(), |r| AstPtr::new(r).wrap_right()),
        );
    }

    Arc::new(res)
}

#[cfg(test)]
mod tests {
    //! This module contains tests for doc-expression parsing.
    //! Currently, it tests `#[doc(hidden)]` and `#[doc(alias)]`.

    use intern::Symbol;
    use span::EditionedFileId;
    use triomphe::Arc;

    use hir_expand::span_map::{RealSpanMap, SpanMap};
    use span::FileId;
    use syntax::{ast, AstNode, TextRange};
    use syntax_bridge::{syntax_node_to_token_tree, DocCommentDesugarMode};

    use crate::attr::{DocAtom, DocExpr};

    fn assert_parse_result(input: &str, expected: DocExpr) {
        let source_file = ast::SourceFile::parse(input, span::Edition::CURRENT).ok().unwrap();
        let tt = source_file.syntax().descendants().find_map(ast::TokenTree::cast).unwrap();
        let map = SpanMap::RealSpanMap(Arc::new(RealSpanMap::absolute(
            EditionedFileId::current_edition(FileId::from_raw(0)),
        )));
        let tt = syntax_node_to_token_tree(
            tt.syntax(),
            map.as_ref(),
            map.span_for_range(TextRange::empty(0.into())),
            DocCommentDesugarMode::ProcMacro,
        );
        let cfg = DocExpr::parse(&tt);
        assert_eq!(cfg, expected);
    }

    #[test]
    fn test_doc_expr_parser() {
        assert_parse_result("#![doc(hidden)]", DocAtom::Flag(Symbol::intern("hidden")).into());

        assert_parse_result(
            r#"#![doc(alias = "foo")]"#,
            DocAtom::KeyValue { key: Symbol::intern("alias"), value: Symbol::intern("foo") }.into(),
        );

        assert_parse_result(
            r#"#![doc(alias("foo"))]"#,
            DocExpr::Alias([Symbol::intern("foo")].into()),
        );
        assert_parse_result(
            r#"#![doc(alias("foo", "bar", "baz"))]"#,
            DocExpr::Alias(
                [Symbol::intern("foo"), Symbol::intern("bar"), Symbol::intern("baz")].into(),
            ),
        );

        assert_parse_result(
            r#"
        #[doc(alias("Bar", "Qux"))]
        struct Foo;"#,
            DocExpr::Alias([Symbol::intern("Bar"), Symbol::intern("Qux")].into()),
        );
    }
}
