//! A higher level attributes based on TokenTree, with also some shortcuts.

use std::{fmt, hash::Hash, ops, sync::Arc};

use base_db::CrateId;
use cfg::{CfgExpr, CfgOptions};
use either::Either;
use hir_expand::{hygiene::Hygiene, name::AsName, HirFileId, InFile};
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
    MacroId, VariantId,
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
impl Attrs {
    pub fn get(&self, id: AttrId) -> Option<&Attr> {
        (**self).iter().find(|attr| attr.id == id)
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
            .filter_map(|(id, attr)| match attr {
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

                let subtree = match attr.token_tree_value() {
                    Some(it) => it,
                    _ => return smallvec![attr.clone()],
                };

                // Input subtree is: `(cfg, $(attr),+)`
                // Split it up into a `cfg` subtree and the `attr` subtrees.
                // FIXME: There should be a common API for this.
                let mut parts = subtree.token_trees.split(|tt| {
                    matches!(tt, tt::TokenTree::Leaf(tt::Leaf::Punct(Punct { char: ',', .. })))
                });
                let cfg = match parts.next() {
                    Some(it) => it,
                    None => return smallvec![],
                };
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
}

impl Attrs {
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
            tt.delimiter_kind() == Some(DelimiterKind::Parenthesis) &&
                matches!(&*tt.token_trees, [tt::TokenTree::Leaf(tt::Leaf::Ident(ident))] if ident.text == "hidden")
        })
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
            AttrDefId::MacroId(it) => match it {
                MacroId::Macro2Id(it) => attrs_from_item_tree(it.lookup(db).id, db),
                MacroId::MacroRulesId(it) => attrs_from_item_tree(it.lookup(db).id, db),
                MacroId::ProcMacroId(it) => attrs_from_item_tree(it.lookup(db).id, db),
            },
            AttrDefId::ImplId(it) => attrs_from_item_tree(it.lookup(db).id, db),
            AttrDefId::ConstId(it) => attrs_from_item_tree(it.lookup(db).id, db),
            AttrDefId::StaticId(it) => attrs_from_item_tree(it.lookup(db).id, db),
            AttrDefId::FunctionId(it) => attrs_from_item_tree(it.lookup(db).id, db),
            AttrDefId::TypeAliasId(it) => attrs_from_item_tree(it.lookup(db).id, db),
            AttrDefId::GenericParamId(it) => match it {
                GenericParamId::ConstParamId(it) => {
                    let src = it.parent().child_source(db);
                    RawAttrs::from_attrs_owner(
                        db,
                        src.with_value(src.value[it.local_id()].as_ref().either(
                            |it| match it {
                                ast::TypeOrConstParam::Type(it) => it as _,
                                ast::TypeOrConstParam::Const(it) => it as _,
                            },
                            |it| it as _,
                        )),
                    )
                }
                GenericParamId::TypeParamId(it) => {
                    let src = it.parent().child_source(db);
                    RawAttrs::from_attrs_owner(
                        db,
                        src.with_value(src.value[it.local_id()].as_ref().either(
                            |it| match it {
                                ast::TypeOrConstParam::Type(it) => it as _,
                                ast::TypeOrConstParam::Const(it) => it as _,
                            },
                            |it| it as _,
                        )),
                    )
                }
                GenericParamId::LifetimeParamId(it) => {
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
            AttrDefId::MacroId(id) => match id {
                MacroId::Macro2Id(id) => id.lookup(db).source(db).map(ast::AnyHasAttrs::new),
                MacroId::MacroRulesId(id) => id.lookup(db).source(db).map(ast::AnyHasAttrs::new),
                MacroId::ProcMacroId(id) => id.lookup(db).source(db).map(ast::AnyHasAttrs::new),
            },
            AttrDefId::ImplId(id) => id.lookup(db).source(db).map(ast::AnyHasAttrs::new),
            AttrDefId::GenericParamId(id) => match id {
                GenericParamId::ConstParamId(id) => {
                    id.parent().child_source(db).map(|source| match &source[id.local_id()] {
                        Either::Left(ast::TypeOrConstParam::Type(id)) => {
                            ast::AnyHasAttrs::new(id.clone())
                        }
                        Either::Left(ast::TypeOrConstParam::Const(id)) => {
                            ast::AnyHasAttrs::new(id.clone())
                        }
                        Either::Right(id) => ast::AnyHasAttrs::new(id.clone()),
                    })
                }
                GenericParamId::TypeParamId(id) => {
                    id.parent().child_source(db).map(|source| match &source[id.local_id()] {
                        Either::Left(ast::TypeOrConstParam::Type(id)) => {
                            ast::AnyHasAttrs::new(id.clone())
                        }
                        Either::Left(ast::TypeOrConstParam::Const(id)) => {
                            ast::AnyHasAttrs::new(id.clone())
                        }
                        Either::Right(id) => ast::AnyHasAttrs::new(id.clone()),
                    })
                }
                GenericParamId::LifetimeParamId(id) => id
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

fn inner_attributes(
    syntax: &SyntaxNode,
) -> Option<impl Iterator<Item = Either<ast::Attr, ast::Comment>>> {
    let node = match_ast! {
        match syntax {
            ast::SourceFile(_) => syntax.clone(),
            ast::ExternBlock(it) => it.extern_item_list()?.syntax().clone(),
            ast::Fn(it) => it.body()?.stmt_list()?.syntax().clone(),
            ast::Impl(it) => it.assoc_item_list()?.syntax().clone(),
            ast::Module(it) => it.item_list()?.syntax().clone(),
            ast::BlockExpr(it) => {
                use syntax::SyntaxKind::{BLOCK_EXPR , EXPR_STMT};
                // Block expressions accept outer and inner attributes, but only when they are the outer
                // expression of an expression statement or the final expression of another block expression.
                let may_carry_attributes = matches!(
                    it.syntax().parent().map(|it| it.kind()),
                     Some(BLOCK_EXPR | EXPR_STMT)
                );
                if !may_carry_attributes {
                    return None
                }
                syntax.clone()
            },
            _ => return None,
        }
    };

    let attrs = ast::AttrDocCommentIter::from_syntax_node(&node).filter(|el| match el {
        Either::Left(attr) => attr.kind().is_inner(),
        Either::Right(comment) => comment.is_inner(),
    });
    Some(attrs)
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
        let path = Interned::new(ModPath::from_src(db.upcast(), ast.path()?, hygiene)?);
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
        let (parse, _) = mbe::token_tree_to_syntax_node(tt, mbe::TopEntryPoint::MetaItem);
        let ast = ast::Meta::cast(parse.syntax_node())?;

        Self::from_src(db, ast, hygiene, id)
    }

    pub fn path(&self) -> &ModPath {
        &self.path
    }
}

impl Attr {
    /// #[path = "string"]
    pub fn string_value(&self) -> Option<&SmolStr> {
        match self.input.as_deref()? {
            AttrInput::Literal(it) => Some(it),
            _ => None,
        }
    }

    /// #[path(ident)]
    pub fn single_ident_value(&self) -> Option<&tt::Ident> {
        match self.input.as_deref()? {
            AttrInput::TokenTree(subtree, _) => match &*subtree.token_trees {
                [tt::TokenTree::Leaf(tt::Leaf::Ident(ident))] => Some(ident),
                _ => None,
            },
            _ => None,
        }
    }

    /// #[path TokenTree]
    pub fn token_tree_value(&self) -> Option<&Subtree> {
        match self.input.as_deref()? {
            AttrInput::TokenTree(subtree, _) => Some(subtree),
            _ => None,
        }
    }

    /// Parses this attribute as a token tree consisting of comma separated paths.
    pub fn parse_path_comma_token_tree(&self) -> Option<impl Iterator<Item = ModPath> + '_> {
        let args = self.token_tree_value()?;

        if args.delimiter_kind() != Some(DelimiterKind::Parenthesis) {
            return None;
        }
        let paths = args
            .token_trees
            .split(|tt| matches!(tt, tt::TokenTree::Leaf(tt::Leaf::Punct(Punct { char: ',', .. }))))
            .map(|tts| {
                let segments = tts.iter().filter_map(|tt| match tt {
                    tt::TokenTree::Leaf(tt::Leaf::Ident(id)) => Some(id.as_name()),
                    _ => None,
                });
                ModPath::from_segments(PathKind::Plain, segments)
            });

        Some(paths)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AttrQuery<'attr> {
    attrs: &'attr Attrs,
    key: &'static str,
}

impl<'attr> AttrQuery<'attr> {
    pub fn tt_values(self) -> impl Iterator<Item = &'attr Subtree> {
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
}

fn attrs_from_item_tree<N: ItemTreeNode>(id: ItemTreeId<N>, db: &dyn DefDatabase) -> RawAttrs {
    let tree = id.item_tree(db);
    let mod_item = N::id_to_mod_item(id.value);
    tree.raw_attrs(mod_item.into()).clone()
}

fn collect_attrs(
    owner: &dyn ast::HasAttrs,
) -> impl Iterator<Item = (AttrId, Either<ast::Attr, ast::Comment>)> {
    let inner_attrs = inner_attributes(owner.syntax()).into_iter().flatten();
    let outer_attrs =
        ast::AttrDocCommentIter::from_syntax_node(owner.syntax()).filter(|el| match el {
            Either::Left(attr) => attr.kind().is_outer(),
            Either::Right(comment) => comment.is_outer(),
        });
    outer_attrs
        .chain(inner_attrs)
        .enumerate()
        .map(|(id, attr)| (AttrId { ast_index: id as u32 }, attr))
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
