//! A higher level attributes based on TokenTree, with also some shortcuts.

pub mod builtin;

#[cfg(test)]
mod tests;

use std::{hash::Hash, ops};

use base_db::CrateId;
use cfg::{CfgExpr, CfgOptions};
use either::Either;
use hir_expand::{
    attrs::{collect_attrs, Attr, AttrId, RawAttrs},
    HirFileId, InFile,
};
use itertools::Itertools;
use la_arena::{ArenaMap, Idx, RawIdx};
use mbe::DelimiterKind;
use syntax::{
    ast::{self, HasAttrs, IsString},
    AstPtr, AstToken, SmolStr, TextRange, TextSize,
};
use triomphe::Arc;

use crate::{
    db::DefDatabase,
    item_tree::{AttrOwner, Fields, ItemTreeId, ItemTreeNode},
    lang_item::LangItem,
    nameres::{ModuleOrigin, ModuleSource},
    src::{HasChildSource, HasSource},
    AdtId, AssocItemLoc, AttrDefId, EnumId, GenericParamId, ItemLoc, LocalEnumVariantId,
    LocalFieldId, Lookup, MacroId, VariantId,
};

/// Holds documentation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Documentation(String);

impl Documentation {
    pub fn new(s: String) -> Self {
        Documentation(s)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<Documentation> for String {
    fn from(Documentation(string): Documentation) -> Self {
        string
    }
}

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

    pub(crate) fn variants_attrs_query(
        db: &dyn DefDatabase,
        e: EnumId,
    ) -> Arc<ArenaMap<LocalEnumVariantId, Attrs>> {
        let _p = profile::span("variants_attrs_query");
        // FIXME: There should be some proper form of mapping between item tree enum variant ids and hir enum variant ids
        let mut res = ArenaMap::default();

        let loc = e.lookup(db);
        let krate = loc.container.krate;
        let item_tree = loc.id.item_tree(db);
        let enum_ = &item_tree[loc.id.value];
        let crate_graph = db.crate_graph();
        let cfg_options = &crate_graph[krate].cfg_options;

        let mut idx = 0;
        for variant in enum_.variants.clone() {
            let attrs = item_tree.attrs(db, krate, variant.into());
            if attrs.is_cfg_enabled(cfg_options) {
                res.insert(Idx::from_raw(RawIdx::from(idx)), attrs);
                idx += 1;
            }
        }

        Arc::new(res)
    }

    pub(crate) fn fields_attrs_query(
        db: &dyn DefDatabase,
        v: VariantId,
    ) -> Arc<ArenaMap<LocalFieldId, Attrs>> {
        let _p = profile::span("fields_attrs_query");
        // FIXME: There should be some proper form of mapping between item tree field ids and hir field ids
        let mut res = ArenaMap::default();

        let crate_graph = db.crate_graph();
        let (fields, item_tree, krate) = match v {
            VariantId::EnumVariantId(it) => {
                let e = it.parent;
                let loc = e.lookup(db);
                let krate = loc.container.krate;
                let item_tree = loc.id.item_tree(db);
                let enum_ = &item_tree[loc.id.value];

                let cfg_options = &crate_graph[krate].cfg_options;

                let Some(variant) = enum_.variants.clone().filter(|variant| {
                    let attrs = item_tree.attrs(db, krate, (*variant).into());
                    attrs.is_cfg_enabled(cfg_options)
                })
                .zip(0u32..)
                .find(|(_variant, idx)| it.local_id == Idx::from_raw(RawIdx::from(*idx)))
                .map(|(variant, _idx)| variant)
                else {
                    return Arc::new(res);
                };

                (item_tree[variant].fields.clone(), item_tree, krate)
            }
            VariantId::StructId(it) => {
                let loc = it.lookup(db);
                let krate = loc.container.krate;
                let item_tree = loc.id.item_tree(db);
                let struct_ = &item_tree[loc.id.value];
                (struct_.fields.clone(), item_tree, krate)
            }
            VariantId::UnionId(it) => {
                let loc = it.lookup(db);
                let krate = loc.container.krate;
                let item_tree = loc.id.item_tree(db);
                let union_ = &item_tree[loc.id.value];
                (union_.fields.clone(), item_tree, krate)
            }
        };

        let fields = match fields {
            Fields::Record(fields) | Fields::Tuple(fields) => fields,
            Fields::Unit => return Arc::new(res),
        };

        let cfg_options = &crate_graph[krate].cfg_options;

        let mut idx = 0;
        for field in fields {
            let attrs = item_tree.attrs(db, krate, field.into());
            if attrs.is_cfg_enabled(cfg_options) {
                res.insert(Idx::from_raw(RawIdx::from(idx)), attrs);
                idx += 1;
            }
        }

        Arc::new(res)
    }
}

impl Attrs {
    pub fn by_key(&self, key: &'static str) -> AttrQuery<'_> {
        AttrQuery { attrs: self, key }
    }

    pub fn cfg(&self) -> Option<CfgExpr> {
        let mut cfgs = self.by_key("cfg").tt_values().map(CfgExpr::parse);
        let first = cfgs.next()?;
        match cfgs.next() {
            Some(second) => {
                let cfgs = [first, second].into_iter().chain(cfgs);
                Some(CfgExpr::All(cfgs.collect()))
            }
            None => Some(first),
        }
    }

    pub(crate) fn is_cfg_enabled(&self, cfg_options: &CfgOptions) -> bool {
        match self.cfg() {
            None => true,
            Some(cfg) => cfg_options.check(&cfg) != Some(false),
        }
    }

    pub fn lang(&self) -> Option<&SmolStr> {
        self.by_key("lang").string_value()
    }

    pub fn lang_item(&self) -> Option<LangItem> {
        self.by_key("lang").string_value().and_then(|it| LangItem::from_str(it))
    }

    pub fn docs(&self) -> Option<Documentation> {
        let docs = self.by_key("doc").attrs().filter_map(|attr| attr.string_value());
        let indent = doc_indent(self);
        let mut buf = String::new();
        for doc in docs {
            // str::lines doesn't yield anything for the empty string
            if !doc.is_empty() {
                buf.extend(Itertools::intersperse(
                    doc.lines().map(|line| {
                        line.char_indices()
                            .nth(indent)
                            .map_or(line, |(offset, _)| &line[offset..])
                            .trim_end()
                    }),
                    "\n",
                ));
            }
            buf.push('\n');
        }
        buf.pop();
        if buf.is_empty() {
            None
        } else {
            Some(Documentation(buf))
        }
    }

    pub fn has_doc_hidden(&self) -> bool {
        self.by_key("doc").tt_values().any(|tt| {
            tt.delimiter.kind == DelimiterKind::Parenthesis &&
                matches!(&*tt.token_trees, [tt::TokenTree::Leaf(tt::Leaf::Ident(ident))] if ident.text == "hidden")
        })
    }

    pub fn doc_exprs(&self) -> impl Iterator<Item = DocExpr> + '_ {
        self.by_key("doc").tt_values().map(DocExpr::parse)
    }

    pub fn doc_aliases(&self) -> impl Iterator<Item = SmolStr> + '_ {
        self.doc_exprs().flat_map(|doc_expr| doc_expr.aliases().to_vec())
    }

    pub fn is_proc_macro(&self) -> bool {
        self.by_key("proc_macro").exists()
    }

    pub fn is_proc_macro_attribute(&self) -> bool {
        self.by_key("proc_macro_attribute").exists()
    }

    pub fn is_proc_macro_derive(&self) -> bool {
        self.by_key("proc_macro_derive").exists()
    }

    pub fn is_unstable(&self) -> bool {
        self.by_key("unstable").exists()
    }
}

use std::slice::Iter as SliceIter;
#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum DocAtom {
    /// eg. `#[doc(hidden)]`
    Flag(SmolStr),
    /// eg. `#[doc(alias = "x")]`
    ///
    /// Note that a key can have multiple values that are all considered "active" at the same time.
    /// For example, `#[doc(alias = "x")]` and `#[doc(alias = "y")]`.
    KeyValue { key: SmolStr, value: SmolStr },
}

// Adapted from `CfgExpr` parsing code
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
// #[cfg_attr(test, derive(derive_arbitrary::Arbitrary))]
pub enum DocExpr {
    Invalid,
    /// eg. `#[doc(hidden)]`, `#[doc(alias = "x")]`
    Atom(DocAtom),
    /// eg. `#[doc(alias("x", "y"))]`
    Alias(Vec<SmolStr>),
}

impl From<DocAtom> for DocExpr {
    fn from(atom: DocAtom) -> Self {
        DocExpr::Atom(atom)
    }
}

impl DocExpr {
    fn parse<S>(tt: &tt::Subtree<S>) -> DocExpr {
        next_doc_expr(&mut tt.token_trees.iter()).unwrap_or(DocExpr::Invalid)
    }

    pub fn aliases(&self) -> &[SmolStr] {
        match self {
            DocExpr::Atom(DocAtom::KeyValue { key, value }) if key == "alias" => {
                std::slice::from_ref(value)
            }
            DocExpr::Alias(aliases) => aliases,
            _ => &[],
        }
    }
}

fn next_doc_expr<S>(it: &mut SliceIter<'_, tt::TokenTree<S>>) -> Option<DocExpr> {
    let name = match it.next() {
        None => return None,
        Some(tt::TokenTree::Leaf(tt::Leaf::Ident(ident))) => ident.text.clone(),
        Some(_) => return Some(DocExpr::Invalid),
    };

    // Peek
    let ret = match it.as_slice().first() {
        Some(tt::TokenTree::Leaf(tt::Leaf::Punct(punct))) if punct.char == '=' => {
            match it.as_slice().get(1) {
                Some(tt::TokenTree::Leaf(tt::Leaf::Literal(literal))) => {
                    it.next();
                    it.next();
                    // FIXME: escape? raw string?
                    let value =
                        SmolStr::new(literal.text.trim_start_matches('"').trim_end_matches('"'));
                    DocAtom::KeyValue { key: name, value }.into()
                }
                _ => return Some(DocExpr::Invalid),
            }
        }
        Some(tt::TokenTree::Subtree(subtree)) => {
            it.next();
            let subs = parse_comma_sep(subtree);
            match name.as_str() {
                "alias" => DocExpr::Alias(subs),
                _ => DocExpr::Invalid,
            }
        }
        _ => DocAtom::Flag(name).into(),
    };

    // Eat comma separator
    if let Some(tt::TokenTree::Leaf(tt::Leaf::Punct(punct))) = it.as_slice().first() {
        if punct.char == ',' {
            it.next();
        }
    }
    Some(ret)
}

fn parse_comma_sep<S>(subtree: &tt::Subtree<S>) -> Vec<SmolStr> {
    subtree
        .token_trees
        .iter()
        .filter_map(|tt| match tt {
            tt::TokenTree::Leaf(tt::Leaf::Literal(lit)) => {
                // FIXME: escape? raw string?
                Some(SmolStr::new(lit.text.trim_start_matches('"').trim_end_matches('"')))
            }
            _ => None,
        })
        .collect()
}

impl AttrsWithOwner {
    pub(crate) fn attrs_with_owner(db: &dyn DefDatabase, owner: AttrDefId) -> Self {
        Self { attrs: db.attrs(owner), owner }
    }

    pub(crate) fn attrs_query(db: &dyn DefDatabase, def: AttrDefId) -> Attrs {
        let _p = profile::span("attrs_query");
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
                    ModuleOrigin::BlockExpr { block } => RawAttrs::from_attrs_owner(
                        db.upcast(),
                        InFile::new(block.file_id, block.to_node(db.upcast()))
                            .as_ref()
                            .map(|it| it as &dyn ast::HasAttrs),
                    ),
                }
            }
            AttrDefId::FieldId(it) => {
                return db.fields_attrs(it.parent)[it.local_id].clone();
            }
            AttrDefId::EnumVariantId(it) => {
                return db.variants_attrs(it.parent)[it.local_id].clone();
            }
            // FIXME: DRY this up
            AttrDefId::AdtId(it) => match it {
                AdtId::StructId(it) => attrs_from_item_tree_loc(db, it),
                AdtId::EnumId(it) => attrs_from_item_tree_loc(db, it),
                AdtId::UnionId(it) => attrs_from_item_tree_loc(db, it),
            },
            AttrDefId::TraitId(it) => attrs_from_item_tree_loc(db, it),
            AttrDefId::TraitAliasId(it) => attrs_from_item_tree_loc(db, it),
            AttrDefId::MacroId(it) => match it {
                MacroId::Macro2Id(it) => attrs_from_item_tree(db, it.lookup(db).id),
                MacroId::MacroRulesId(it) => attrs_from_item_tree(db, it.lookup(db).id),
                MacroId::ProcMacroId(it) => attrs_from_item_tree(db, it.lookup(db).id),
            },
            AttrDefId::ImplId(it) => attrs_from_item_tree_loc(db, it),
            AttrDefId::ConstId(it) => attrs_from_item_tree_assoc(db, it),
            AttrDefId::StaticId(it) => attrs_from_item_tree_assoc(db, it),
            AttrDefId::FunctionId(it) => attrs_from_item_tree_assoc(db, it),
            AttrDefId::TypeAliasId(it) => attrs_from_item_tree_assoc(db, it),
            AttrDefId::GenericParamId(it) => match it {
                GenericParamId::ConstParamId(it) => {
                    let src = it.parent().child_source(db);
                    RawAttrs::from_attrs_owner(
                        db.upcast(),
                        src.with_value(&src.value[it.local_id()]),
                    )
                }
                GenericParamId::TypeParamId(it) => {
                    let src = it.parent().child_source(db);
                    RawAttrs::from_attrs_owner(
                        db.upcast(),
                        src.with_value(&src.value[it.local_id()]),
                    )
                }
                GenericParamId::LifetimeParamId(it) => {
                    let src = it.parent.child_source(db);
                    RawAttrs::from_attrs_owner(db.upcast(), src.with_value(&src.value[it.local_id]))
                }
            },
            AttrDefId::ExternBlockId(it) => attrs_from_item_tree_loc(db, it),
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
                let owner = match &map[id.local_id] {
                    Either::Left(it) => ast::AnyHasAttrs::new(it.to_node(&root)),
                    Either::Right(it) => ast::AnyHasAttrs::new(it.to_node(&root)),
                };
                InFile::new(file_id, owner)
            }
            AttrDefId::AdtId(adt) => match adt {
                AdtId::StructId(id) => any_has_attrs(db, id),
                AdtId::UnionId(id) => any_has_attrs(db, id),
                AdtId::EnumId(id) => any_has_attrs(db, id),
            },
            AttrDefId::FunctionId(id) => any_has_attrs(db, id),
            AttrDefId::EnumVariantId(id) => {
                let map = db.variants_attrs_source_map(id.parent);
                let file_id = id.parent.lookup(db).id.file_id();
                let root = db.parse_or_expand(file_id);
                InFile::new(file_id, ast::AnyHasAttrs::new(map[id.local_id].to_node(&root)))
            }
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
        };

        AttrSourceMap::new(owner.as_ref().map(|node| node as &dyn HasAttrs))
    }

    pub fn docs_with_rangemap(
        &self,
        db: &dyn DefDatabase,
    ) -> Option<(Documentation, DocsRangeMap)> {
        let docs =
            self.by_key("doc").attrs().filter_map(|attr| attr.string_value().map(|s| (s, attr.id)));
        let indent = doc_indent(self);
        let mut buf = String::new();
        let mut mapping = Vec::new();
        for (doc, idx) in docs {
            if !doc.is_empty() {
                let mut base_offset = 0;
                for raw_line in doc.split('\n') {
                    let line = raw_line.trim_end();
                    let line_len = line.len();
                    let (offset, line) = match line.char_indices().nth(indent) {
                        Some((offset, _)) => (offset, &line[offset..]),
                        None => (0, line),
                    };
                    let buf_offset = buf.len();
                    buf.push_str(line);
                    mapping.push((
                        TextRange::new(buf_offset.try_into().ok()?, buf.len().try_into().ok()?),
                        idx,
                        TextRange::at(
                            (base_offset + offset).try_into().ok()?,
                            line_len.try_into().ok()?,
                        ),
                    ));
                    buf.push('\n');
                    base_offset += raw_line.len() + 1;
                }
            } else {
                buf.push('\n');
            }
        }
        buf.pop();
        if buf.is_empty() {
            None
        } else {
            Some((Documentation(buf), DocsRangeMap { mapping, source_map: self.source_map(db) }))
        }
    }
}

fn doc_indent(attrs: &Attrs) -> usize {
    attrs
        .by_key("doc")
        .attrs()
        .filter_map(|attr| attr.string_value())
        .flat_map(|s| s.lines())
        .filter(|line| !line.chars().all(|c| c.is_whitespace()))
        .map(|line| line.chars().take_while(|c| c.is_whitespace()).count())
        .min()
        .unwrap_or(0)
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

    fn source_of_id(&self, id: AttrId) -> InFile<&Either<ast::Attr, ast::Comment>> {
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

/// A struct to map text ranges from [`Documentation`] back to TextRanges in the syntax tree.
#[derive(Debug)]
pub struct DocsRangeMap {
    source_map: AttrSourceMap,
    // (docstring-line-range, attr_index, attr-string-range)
    // a mapping from the text range of a line of the [`Documentation`] to the attribute index and
    // the original (untrimmed) syntax doc line
    mapping: Vec<(TextRange, AttrId, TextRange)>,
}

impl DocsRangeMap {
    /// Maps a [`TextRange`] relative to the documentation string back to its AST range
    pub fn map(&self, range: TextRange) -> Option<InFile<TextRange>> {
        let found = self.mapping.binary_search_by(|(probe, ..)| probe.ordering(range)).ok()?;
        let (line_docs_range, idx, original_line_src_range) = self.mapping[found];
        if !line_docs_range.contains_range(range) {
            return None;
        }

        let relative_range = range - line_docs_range.start();

        let InFile { file_id, value: source } = self.source_map.source_of_id(idx);
        match source {
            Either::Left(attr) => {
                let string = get_doc_string_in_attr(attr)?;
                let text_range = string.open_quote_text_range()?;
                let range = TextRange::at(
                    text_range.end() + original_line_src_range.start() + relative_range.start(),
                    string.syntax().text_range().len().min(range.len()),
                );
                Some(InFile { file_id, value: range })
            }
            Either::Right(comment) => {
                let text_range = comment.syntax().text_range();
                let range = TextRange::at(
                    text_range.start()
                        + TextSize::try_from(comment.prefix().len()).ok()?
                        + original_line_src_range.start()
                        + relative_range.start(),
                    text_range.len().min(range.len()),
                );
                Some(InFile { file_id, value: range })
            }
        }
    }
}

fn get_doc_string_in_attr(it: &ast::Attr) -> Option<ast::String> {
    match it.expr() {
        // #[doc = lit]
        Some(ast::Expr::Literal(lit)) => match lit.kind() {
            ast::LiteralKind::String(it) => Some(it),
            _ => None,
        },
        // #[cfg_attr(..., doc = "", ...)]
        None => {
            // FIXME: See highlight injection for what to do here
            None
        }
        _ => None,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AttrQuery<'attr> {
    attrs: &'attr Attrs,
    key: &'static str,
}

impl<'attr> AttrQuery<'attr> {
    pub fn tt_values(self) -> impl Iterator<Item = &'attr crate::tt::Subtree> {
        self.attrs().filter_map(|attr| attr.token_tree_value())
    }

    pub fn string_value(self) -> Option<&'attr SmolStr> {
        self.attrs().find_map(|attr| attr.string_value())
    }

    pub fn exists(self) -> bool {
        self.attrs().next().is_some()
    }

    pub fn attrs(self) -> impl Iterator<Item = &'attr Attr> + Clone {
        let key = self.key;
        self.attrs
            .iter()
            .filter(move |attr| attr.path.as_ident().map_or(false, |s| s.to_smol_str() == key))
    }

    /// Find string value for a specific key inside token tree
    ///
    /// ```ignore
    /// #[doc(html_root_url = "url")]
    ///       ^^^^^^^^^^^^^ key
    /// ```
    pub fn find_string_value_in_tt(self, key: &'attr str) -> Option<&SmolStr> {
        self.tt_values().find_map(|tt| {
            let name = tt.token_trees.iter()
                .skip_while(|tt| !matches!(tt, tt::TokenTree::Leaf(tt::Leaf::Ident(tt::Ident { text, ..} )) if text == key))
                .nth(2);

            match name {
                Some(tt::TokenTree::Leaf(tt::Leaf::Literal(tt::Literal{ ref text, ..}))) => Some(text),
                _ => None
            }
        })
    }
}

fn any_has_attrs(
    db: &dyn DefDatabase,
    id: impl Lookup<Data = impl HasSource<Value = impl ast::HasAttrs>>,
) -> InFile<ast::AnyHasAttrs> {
    id.lookup(db).source(db).map(ast::AnyHasAttrs::new)
}

fn attrs_from_item_tree<N: ItemTreeNode>(db: &dyn DefDatabase, id: ItemTreeId<N>) -> RawAttrs {
    let tree = id.item_tree(db);
    let mod_item = N::id_to_mod_item(id.value);
    tree.raw_attrs(mod_item.into()).clone()
}

fn attrs_from_item_tree_loc<N: ItemTreeNode>(
    db: &dyn DefDatabase,
    lookup: impl Lookup<Data = ItemLoc<N>>,
) -> RawAttrs {
    let id = lookup.lookup(db).id;
    attrs_from_item_tree(db, id)
}

fn attrs_from_item_tree_assoc<N: ItemTreeNode>(
    db: &dyn DefDatabase,
    lookup: impl Lookup<Data = AssocItemLoc<N>>,
) -> RawAttrs {
    let id = lookup.lookup(db).id;
    attrs_from_item_tree(db, id)
}

pub(crate) fn variants_attrs_source_map(
    db: &dyn DefDatabase,
    def: EnumId,
) -> Arc<ArenaMap<LocalEnumVariantId, AstPtr<ast::Variant>>> {
    let mut res = ArenaMap::default();
    let child_source = def.child_source(db);

    for (idx, variant) in child_source.value.iter() {
        res.insert(idx, AstPtr::new(variant));
    }

    Arc::new(res)
}

pub(crate) fn fields_attrs_source_map(
    db: &dyn DefDatabase,
    def: VariantId,
) -> Arc<ArenaMap<LocalFieldId, Either<AstPtr<ast::TupleField>, AstPtr<ast::RecordField>>>> {
    let mut res = ArenaMap::default();
    let child_source = def.child_source(db);

    for (idx, variant) in child_source.value.iter() {
        res.insert(
            idx,
            variant
                .as_ref()
                .either(|l| Either::Left(AstPtr::new(l)), |r| Either::Right(AstPtr::new(r))),
        );
    }

    Arc::new(res)
}
