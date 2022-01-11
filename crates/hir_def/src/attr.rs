//! A higher level attributes based on TokenTree, with also some shortcuts.

use std::{fmt, hash::Hash, ops, sync::Arc};

use base_db::CrateId;
use cfg::{CfgExpr, CfgOptions};
use either::Either;
use hir_expand::{hygiene::Hygiene, name::AsName, AstId, HirFileId, InFile};
use itertools::Itertools;
use la_arena::ArenaMap;
use mbe::{syntax_node_to_token_tree, DelimiterKind, Punct};
use smallvec::{smallvec, SmallVec};
use syntax::{
    ast::{self, AstNode, HasAttrs, IsString},
    match_ast, AstPtr, AstToken, SmolStr, SyntaxNode, TextRange, TextSize,
};
use tt::Subtree;

use crate::{
    db::DefDatabase,
    intern::Interned,
    item_tree::{ItemTreeId, ItemTreeNode},
    nameres::ModuleSource,
    path::{ModPath, PathKind},
    src::{HasChildSource, HasSource},
    AdtId, AttrDefId, EnumId, GenericParamId, HasModule, LocalEnumVariantId, LocalFieldId, Lookup,
    VariantId,
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

/// Syntactical attributes, without filtering of `cfg_attr`s.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub(crate) struct RawAttrs {
    entries: Option<Arc<[Attr]>>,
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct Attrs(RawAttrs);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttrsWithOwner {
    attrs: Attrs,
    owner: AttrDefId,
}

impl ops::Deref for RawAttrs {
    type Target = [Attr];

    fn deref(&self) -> &[Attr] {
        match &self.entries {
            Some(it) => &*it,
            None => &[],
        }
    }
}

impl ops::Deref for Attrs {
    type Target = [Attr];

    fn deref(&self) -> &[Attr] {
        match &self.0.entries {
            Some(it) => &*it,
            None => &[],
        }
    }
}

impl ops::Index<AttrId> for Attrs {
    type Output = Attr;

    fn index(&self, AttrId { ast_index, .. }: AttrId) -> &Self::Output {
        &(**self)[ast_index as usize]
    }
}

impl ops::Deref for AttrsWithOwner {
    type Target = Attrs;

    fn deref(&self) -> &Attrs {
        &self.attrs
    }
}

impl RawAttrs {
    pub(crate) const EMPTY: Self = Self { entries: None };

    pub(crate) fn new(db: &dyn DefDatabase, owner: &dyn ast::HasAttrs, hygiene: &Hygiene) -> Self {
        let entries = collect_attrs(owner)
            .flat_map(|(id, attr)| match attr {
                Either::Left(attr) => {
                    attr.meta().and_then(|meta| Attr::from_src(db, meta, hygiene, id))
                }
                Either::Right(comment) => comment.doc_comment().map(|doc| Attr {
                    id,
                    input: Some(Interned::new(AttrInput::Literal(SmolStr::new(doc)))),
                    path: Interned::new(ModPath::from(hir_expand::name!(doc))),
                }),
            })
            .collect::<Arc<_>>();

        Self { entries: if entries.is_empty() { None } else { Some(entries) } }
    }

    fn from_attrs_owner(db: &dyn DefDatabase, owner: InFile<&dyn ast::HasAttrs>) -> Self {
        let hygiene = Hygiene::new(db.upcast(), owner.file_id);
        Self::new(db, owner.value, &hygiene)
    }

    pub(crate) fn merge(&self, other: Self) -> Self {
        // FIXME: This needs to fixup `AttrId`s
        match (&self.entries, &other.entries) {
            (None, None) => Self::EMPTY,
            (Some(entries), None) | (None, Some(entries)) => {
                Self { entries: Some(entries.clone()) }
            }
            (Some(a), Some(b)) => {
                Self { entries: Some(a.iter().chain(b.iter()).cloned().collect()) }
            }
        }
    }

    /// Processes `cfg_attr`s, returning the resulting semantic `Attrs`.
    pub(crate) fn filter(self, db: &dyn DefDatabase, krate: CrateId) -> Attrs {
        let has_cfg_attrs = self.iter().any(|attr| {
            attr.path.as_ident().map_or(false, |name| *name == hir_expand::name![cfg_attr])
        });
        if !has_cfg_attrs {
            return Attrs(self);
        }

        let crate_graph = db.crate_graph();
        let new_attrs = self
            .iter()
            .flat_map(|attr| -> SmallVec<[_; 1]> {
                let is_cfg_attr =
                    attr.path.as_ident().map_or(false, |name| *name == hir_expand::name![cfg_attr]);
                if !is_cfg_attr {
                    return smallvec![attr.clone()];
                }

                let subtree = match attr.input.as_deref() {
                    Some(AttrInput::TokenTree(it, _)) => it,
                    _ => return smallvec![attr.clone()],
                };

                // Input subtree is: `(cfg, $(attr),+)`
                // Split it up into a `cfg` subtree and the `attr` subtrees.
                // FIXME: There should be a common API for this.
                let mut parts = subtree.token_trees.split(
                    |tt| matches!(tt, tt::TokenTree::Leaf(tt::Leaf::Punct(p)) if p.char == ','),
                );
                let cfg = parts.next().unwrap();
                let cfg = Subtree { delimiter: subtree.delimiter, token_trees: cfg.to_vec() };
                let cfg = CfgExpr::parse(&cfg);
                let index = attr.id;
                let attrs = parts.filter(|a| !a.is_empty()).filter_map(|attr| {
                    let tree = Subtree { delimiter: None, token_trees: attr.to_vec() };
                    // FIXME hygiene
                    let hygiene = Hygiene::new_unhygienic();
                    Attr::from_tt(db, &tree, &hygiene, index)
                });

                let cfg_options = &crate_graph[krate].cfg_options;
                if cfg_options.check(&cfg) == Some(false) {
                    smallvec![]
                } else {
                    cov_mark::hit!(cfg_attr_active);

                    attrs.collect()
                }
            })
            .collect();

        Attrs(RawAttrs { entries: Some(new_attrs) })
    }
}

impl Attrs {
    pub const EMPTY: Self = Self(RawAttrs::EMPTY);

    pub(crate) fn variants_attrs_query(
        db: &dyn DefDatabase,
        e: EnumId,
    ) -> Arc<ArenaMap<LocalEnumVariantId, Attrs>> {
        let krate = e.lookup(db).container.krate;
        let src = e.child_source(db);
        let mut res = ArenaMap::default();

        for (id, var) in src.value.iter() {
            let attrs = RawAttrs::from_attrs_owner(db, src.with_value(var as &dyn ast::HasAttrs))
                .filter(db, krate);

            res.insert(id, attrs)
        }

        Arc::new(res)
    }

    pub(crate) fn fields_attrs_query(
        db: &dyn DefDatabase,
        v: VariantId,
    ) -> Arc<ArenaMap<LocalFieldId, Attrs>> {
        let krate = v.module(db).krate;
        let src = v.child_source(db);
        let mut res = ArenaMap::default();

        for (id, fld) in src.value.iter() {
            let owner: &dyn HasAttrs = match fld {
                Either::Left(tuple) => tuple,
                Either::Right(record) => record,
            };
            let attrs = RawAttrs::from_attrs_owner(db, src.with_value(owner)).filter(db, krate);

            res.insert(id, attrs);
        }

        Arc::new(res)
    }

    pub fn by_key(&self, key: &'static str) -> AttrQuery<'_> {
        AttrQuery { attrs: self, key }
    }

    pub fn cfg(&self) -> Option<CfgExpr> {
        let mut cfgs = self.by_key("cfg").tt_values().map(CfgExpr::parse).collect::<Vec<_>>();
        match cfgs.len() {
            0 => None,
            1 => Some(cfgs.pop().unwrap()),
            _ => Some(CfgExpr::All(cfgs)),
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

    pub fn docs(&self) -> Option<Documentation> {
        let docs = self.by_key("doc").attrs().flat_map(|attr| match attr.input.as_deref()? {
            AttrInput::Literal(s) => Some(s),
            AttrInput::TokenTree(..) => None,
        });
        let indent = docs
            .clone()
            .flat_map(|s| s.lines())
            .filter(|line| !line.chars().all(|c| c.is_whitespace()))
            .map(|line| line.chars().take_while(|c| c.is_whitespace()).count())
            .min()
            .unwrap_or(0);
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
            tt.delimiter_kind() == Some(DelimiterKind::Parenthesis) &&
                matches!(&*tt.token_trees, [tt::TokenTree::Leaf(tt::Leaf::Ident(ident))] if ident.text == "hidden")
        })
    }
}

impl AttrsWithOwner {
    pub(crate) fn attrs_query(db: &dyn DefDatabase, def: AttrDefId) -> Self {
        // FIXME: this should use `Trace` to avoid duplication in `source_map` below
        let raw_attrs = match def {
            AttrDefId::ModuleId(module) => {
                let def_map = module.def_map(db);
                let mod_data = &def_map[module.local_id];
                match mod_data.declaration_source(db) {
                    Some(it) => {
                        let raw_attrs = RawAttrs::from_attrs_owner(
                            db,
                            it.as_ref().map(|it| it as &dyn ast::HasAttrs),
                        );
                        match mod_data.definition_source(db) {
                            InFile { file_id, value: ModuleSource::SourceFile(file) } => raw_attrs
                                .merge(RawAttrs::from_attrs_owner(db, InFile::new(file_id, &file))),
                            _ => raw_attrs,
                        }
                    }
                    None => RawAttrs::from_attrs_owner(
                        db,
                        mod_data.definition_source(db).as_ref().map(|src| match src {
                            ModuleSource::SourceFile(file) => file as &dyn ast::HasAttrs,
                            ModuleSource::Module(module) => module as &dyn ast::HasAttrs,
                            ModuleSource::BlockExpr(block) => block as &dyn ast::HasAttrs,
                        }),
                    ),
                }
            }
            AttrDefId::FieldId(it) => {
                return Self { attrs: db.fields_attrs(it.parent)[it.local_id].clone(), owner: def };
            }
            AttrDefId::EnumVariantId(it) => {
                return Self {
                    attrs: db.variants_attrs(it.parent)[it.local_id].clone(),
                    owner: def,
                };
            }
            AttrDefId::AdtId(it) => match it {
                AdtId::StructId(it) => attrs_from_item_tree(it.lookup(db).id, db),
                AdtId::EnumId(it) => attrs_from_item_tree(it.lookup(db).id, db),
                AdtId::UnionId(it) => attrs_from_item_tree(it.lookup(db).id, db),
            },
            AttrDefId::TraitId(it) => attrs_from_item_tree(it.lookup(db).id, db),
            AttrDefId::MacroDefId(it) => it
                .ast_id()
                .either(|ast_id| attrs_from_ast(ast_id, db), |ast_id| attrs_from_ast(ast_id, db)),
            AttrDefId::ImplId(it) => attrs_from_item_tree(it.lookup(db).id, db),
            AttrDefId::ConstId(it) => attrs_from_item_tree(it.lookup(db).id, db),
            AttrDefId::StaticId(it) => attrs_from_item_tree(it.lookup(db).id, db),
            AttrDefId::FunctionId(it) => attrs_from_item_tree(it.lookup(db).id, db),
            AttrDefId::TypeAliasId(it) => attrs_from_item_tree(it.lookup(db).id, db),
            AttrDefId::GenericParamId(it) => match it {
                GenericParamId::TypeParamId(it) => {
                    let src = it.parent.child_source(db);
                    RawAttrs::from_attrs_owner(
                        db,
                        src.with_value(
                            src.value[it.local_id].as_ref().either(|it| it as _, |it| it as _),
                        ),
                    )
                }
                GenericParamId::LifetimeParamId(it) => {
                    let src = it.parent.child_source(db);
                    RawAttrs::from_attrs_owner(db, src.with_value(&src.value[it.local_id]))
                }
                GenericParamId::ConstParamId(it) => {
                    let src = it.parent.child_source(db);
                    RawAttrs::from_attrs_owner(db, src.with_value(&src.value[it.local_id]))
                }
            },
            AttrDefId::ExternBlockId(it) => attrs_from_item_tree(it.lookup(db).id, db),
        };

        let attrs = raw_attrs.filter(db, def.krate(db));
        Self { attrs, owner: def }
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
                let root = db.parse_or_expand(file_id).unwrap();
                let owner = match &map[id.local_id] {
                    Either::Left(it) => ast::AnyHasAttrs::new(it.to_node(&root)),
                    Either::Right(it) => ast::AnyHasAttrs::new(it.to_node(&root)),
                };
                InFile::new(file_id, owner)
            }
            AttrDefId::AdtId(adt) => match adt {
                AdtId::StructId(id) => id.lookup(db).source(db).map(ast::AnyHasAttrs::new),
                AdtId::UnionId(id) => id.lookup(db).source(db).map(ast::AnyHasAttrs::new),
                AdtId::EnumId(id) => id.lookup(db).source(db).map(ast::AnyHasAttrs::new),
            },
            AttrDefId::FunctionId(id) => id.lookup(db).source(db).map(ast::AnyHasAttrs::new),
            AttrDefId::EnumVariantId(id) => {
                let map = db.variants_attrs_source_map(id.parent);
                let file_id = id.parent.lookup(db).id.file_id();
                let root = db.parse_or_expand(file_id).unwrap();
                InFile::new(file_id, ast::AnyHasAttrs::new(map[id.local_id].to_node(&root)))
            }
            AttrDefId::StaticId(id) => id.lookup(db).source(db).map(ast::AnyHasAttrs::new),
            AttrDefId::ConstId(id) => id.lookup(db).source(db).map(ast::AnyHasAttrs::new),
            AttrDefId::TraitId(id) => id.lookup(db).source(db).map(ast::AnyHasAttrs::new),
            AttrDefId::TypeAliasId(id) => id.lookup(db).source(db).map(ast::AnyHasAttrs::new),
            AttrDefId::MacroDefId(id) => id.ast_id().either(
                |it| it.with_value(ast::AnyHasAttrs::new(it.to_node(db.upcast()))),
                |it| it.with_value(ast::AnyHasAttrs::new(it.to_node(db.upcast()))),
            ),
            AttrDefId::ImplId(id) => id.lookup(db).source(db).map(ast::AnyHasAttrs::new),
            AttrDefId::GenericParamId(id) => match id {
                GenericParamId::TypeParamId(id) => {
                    id.parent.child_source(db).map(|source| match &source[id.local_id] {
                        Either::Left(id) => ast::AnyHasAttrs::new(id.clone()),
                        Either::Right(id) => ast::AnyHasAttrs::new(id.clone()),
                    })
                }
                GenericParamId::LifetimeParamId(id) => id
                    .parent
                    .child_source(db)
                    .map(|source| ast::AnyHasAttrs::new(source[id.local_id].clone())),
                GenericParamId::ConstParamId(id) => id
                    .parent
                    .child_source(db)
                    .map(|source| ast::AnyHasAttrs::new(source[id.local_id].clone())),
            },
            AttrDefId::ExternBlockId(id) => id.lookup(db).source(db).map(ast::AnyHasAttrs::new),
        };

        AttrSourceMap::new(owner.as_ref().map(|node| node as &dyn HasAttrs))
    }

    pub fn docs_with_rangemap(
        &self,
        db: &dyn DefDatabase,
    ) -> Option<(Documentation, DocsRangeMap)> {
        // FIXME: code duplication in `docs` above
        let docs = self.by_key("doc").attrs().flat_map(|attr| match attr.input.as_deref()? {
            AttrInput::Literal(s) => Some((s, attr.id)),
            AttrInput::TokenTree(..) => None,
        });
        let indent = docs
            .clone()
            .flat_map(|(s, _)| s.lines())
            .filter(|line| !line.chars().all(|c| c.is_whitespace()))
            .map(|line| line.chars().take_while(|c| c.is_whitespace()).count())
            .min()
            .unwrap_or(0);
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

fn inner_attributes(
    syntax: &SyntaxNode,
) -> Option<(impl Iterator<Item = ast::Attr>, impl Iterator<Item = ast::Comment>)> {
    let (attrs, docs) = match_ast! {
        match syntax {
            ast::SourceFile(it) => (it.attrs(), ast::DocCommentIter::from_syntax_node(it.syntax())),
            ast::ExternBlock(it) => {
                let extern_item_list = it.extern_item_list()?;
                (extern_item_list.attrs(), ast::DocCommentIter::from_syntax_node(extern_item_list.syntax()))
            },
            ast::Fn(it) => {
                let body = it.body()?;
                let stmt_list = body.stmt_list()?;
                (stmt_list.attrs(), ast::DocCommentIter::from_syntax_node(body.syntax()))
            },
            ast::Impl(it) => {
                let assoc_item_list = it.assoc_item_list()?;
                (assoc_item_list.attrs(), ast::DocCommentIter::from_syntax_node(assoc_item_list.syntax()))
            },
            ast::Module(it) => {
                let item_list = it.item_list()?;
                (item_list.attrs(), ast::DocCommentIter::from_syntax_node(item_list.syntax()))
            },
            // FIXME: BlockExpr's only accept inner attributes in specific cases
            // Excerpt from the reference:
            // Block expressions accept outer and inner attributes, but only when they are the outer
            // expression of an expression statement or the final expression of another block expression.
            ast::BlockExpr(_it) => return None,
            _ => return None,
        }
    };
    let attrs = attrs.filter(|attr| attr.kind().is_inner());
    let docs = docs.filter(|doc| doc.is_inner());
    Some((attrs, docs))
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
        let ast_idx = id.ast_index as usize;
        let file_id = match self.mod_def_site_file_id {
            Some((file_id, def_site_cut)) if def_site_cut <= ast_idx => file_id,
            _ => self.file_id,
        };

        self.source
            .get(ast_idx)
            .map(|it| InFile::new(file_id, it))
            .unwrap_or_else(|| panic!("cannot find attr at index {:?}", id))
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
                let string = get_doc_string_in_attr(&attr)?;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AttrId {
    pub(crate) ast_index: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attr {
    pub(crate) id: AttrId,
    pub(crate) path: Interned<ModPath>,
    pub(crate) input: Option<Interned<AttrInput>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AttrInput {
    /// `#[attr = "string"]`
    Literal(SmolStr),
    /// `#[attr(subtree)]`
    TokenTree(tt::Subtree, mbe::TokenMap),
}

impl fmt::Display for AttrInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AttrInput::Literal(lit) => write!(f, " = \"{}\"", lit.escape_debug()),
            AttrInput::TokenTree(subtree, _) => subtree.fmt(f),
        }
    }
}

impl Attr {
    fn from_src(
        db: &dyn DefDatabase,
        ast: ast::Meta,
        hygiene: &Hygiene,
        id: AttrId,
    ) -> Option<Attr> {
        let path = Interned::new(ModPath::from_src(db, ast.path()?, hygiene)?);
        let input = if let Some(ast::Expr::Literal(lit)) = ast.expr() {
            let value = match lit.kind() {
                ast::LiteralKind::String(string) => string.value()?.into(),
                _ => lit.syntax().first_token()?.text().trim_matches('"').into(),
            };
            Some(Interned::new(AttrInput::Literal(value)))
        } else if let Some(tt) = ast.token_tree() {
            let (tree, map) = syntax_node_to_token_tree(tt.syntax());
            Some(Interned::new(AttrInput::TokenTree(tree, map)))
        } else {
            None
        };
        Some(Attr { id, path, input })
    }

    fn from_tt(
        db: &dyn DefDatabase,
        tt: &tt::Subtree,
        hygiene: &Hygiene,
        id: AttrId,
    ) -> Option<Attr> {
        let (parse, _) = mbe::token_tree_to_syntax_node(tt, mbe::TopEntryPoint::MetaItem).ok()?;
        let ast = ast::Meta::cast(parse.syntax_node())?;

        Self::from_src(db, ast, hygiene, id)
    }

    /// Parses this attribute as a token tree consisting of comma separated paths.
    pub fn parse_path_comma_token_tree(&self) -> Option<impl Iterator<Item = ModPath> + '_> {
        let args = match self.input.as_deref() {
            Some(AttrInput::TokenTree(args, _)) => args,
            _ => return None,
        };

        if args.delimiter_kind() != Some(DelimiterKind::Parenthesis) {
            return None;
        }
        let paths = args
            .token_trees
            .iter()
            .group_by(|tt| {
                matches!(tt, tt::TokenTree::Leaf(tt::Leaf::Punct(Punct { char: ',', .. })))
            })
            .into_iter()
            .filter(|(comma, _)| !*comma)
            .map(|(_, tts)| {
                let segments = tts.filter_map(|tt| match tt {
                    tt::TokenTree::Leaf(tt::Leaf::Ident(id)) => Some(id.as_name()),
                    _ => None,
                });
                ModPath::from_segments(PathKind::Plain, segments)
            })
            .collect::<Vec<_>>();

        Some(paths.into_iter())
    }

    pub fn path(&self) -> &ModPath {
        &self.path
    }

    pub fn string_value(&self) -> Option<&SmolStr> {
        match self.input.as_deref()? {
            AttrInput::Literal(it) => Some(it),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AttrQuery<'attr> {
    attrs: &'attr Attrs,
    key: &'static str,
}

impl<'attr> AttrQuery<'attr> {
    pub fn tt_values(self) -> impl Iterator<Item = &'attr Subtree> {
        self.attrs().filter_map(|attr| match attr.input.as_deref()? {
            AttrInput::TokenTree(it, _) => Some(it),
            _ => None,
        })
    }

    pub fn string_value(self) -> Option<&'attr SmolStr> {
        self.attrs().find_map(|attr| match attr.input.as_deref()? {
            AttrInput::Literal(it) => Some(it),
            _ => None,
        })
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
}

fn attrs_from_ast<N>(src: AstId<N>, db: &dyn DefDatabase) -> RawAttrs
where
    N: ast::HasAttrs,
{
    let src = InFile::new(src.file_id, src.to_node(db.upcast()));
    RawAttrs::from_attrs_owner(db, src.as_ref().map(|it| it as &dyn ast::HasAttrs))
}

fn attrs_from_item_tree<N: ItemTreeNode>(id: ItemTreeId<N>, db: &dyn DefDatabase) -> RawAttrs {
    let tree = id.item_tree(db);
    let mod_item = N::id_to_mod_item(id.value);
    tree.raw_attrs(mod_item.into()).clone()
}

fn collect_attrs(
    owner: &dyn ast::HasAttrs,
) -> impl Iterator<Item = (AttrId, Either<ast::Attr, ast::Comment>)> {
    let (inner_attrs, inner_docs) = inner_attributes(owner.syntax())
        .map_or((None, None), |(attrs, docs)| (Some(attrs), Some(docs)));

    let outer_attrs = owner.attrs().filter(|attr| attr.kind().is_outer());
    let attrs = outer_attrs
        .chain(inner_attrs.into_iter().flatten())
        .map(|attr| (attr.syntax().text_range().start(), Either::Left(attr)));

    let outer_docs =
        ast::DocCommentIter::from_syntax_node(owner.syntax()).filter(ast::Comment::is_outer);
    let docs = outer_docs
        .chain(inner_docs.into_iter().flatten())
        .map(|docs_text| (docs_text.syntax().text_range().start(), Either::Right(docs_text)));
    // sort here by syntax node offset because the source can have doc attributes and doc strings be interleaved
    docs.chain(attrs)
        .sorted_by_key(|&(offset, _)| offset)
        .enumerate()
        .map(|(id, (_, attr))| (AttrId { ast_index: id as u32 }, attr))
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
