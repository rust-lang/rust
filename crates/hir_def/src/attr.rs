//! A higher level attributes based on TokenTree, with also some shortcuts.

use std::{
    convert::{TryFrom, TryInto},
    ops,
    sync::Arc,
};

use base_db::CrateId;
use cfg::{CfgExpr, CfgOptions};
use either::Either;
use hir_expand::{hygiene::Hygiene, name::AsName, AstId, InFile};
use itertools::Itertools;
use la_arena::ArenaMap;
use mbe::ast_to_token_tree;
use smallvec::{smallvec, SmallVec};
use syntax::{
    ast::{self, AstNode, AttrsOwner},
    match_ast, AstToken, SmolStr, SyntaxNode, TextRange, TextSize,
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

impl ops::Deref for AttrsWithOwner {
    type Target = Attrs;

    fn deref(&self) -> &Attrs {
        &self.attrs
    }
}

impl RawAttrs {
    pub(crate) const EMPTY: Self = Self { entries: None };

    pub(crate) fn new(owner: &dyn ast::AttrsOwner, hygiene: &Hygiene) -> Self {
        let entries = collect_attrs(owner)
            .enumerate()
            .flat_map(|(i, attr)| match attr {
                Either::Left(attr) => Attr::from_src(attr, hygiene, i as u32),
                Either::Right(comment) => comment.doc_comment().map(|doc| Attr {
                    index: i as u32,
                    input: Some(AttrInput::Literal(SmolStr::new(doc))),
                    path: Interned::new(ModPath::from(hir_expand::name!(doc))),
                }),
            })
            .collect::<Arc<_>>();

        Self { entries: if entries.is_empty() { None } else { Some(entries) } }
    }

    fn from_attrs_owner(db: &dyn DefDatabase, owner: InFile<&dyn ast::AttrsOwner>) -> Self {
        let hygiene = Hygiene::new(db.upcast(), owner.file_id);
        Self::new(owner.value, &hygiene)
    }

    pub(crate) fn merge(&self, other: Self) -> Self {
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

                let subtree = match &attr.input {
                    Some(AttrInput::TokenTree(it)) => it,
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
                let index = attr.index;
                let attrs = parts.filter(|a| !a.is_empty()).filter_map(|attr| {
                    let tree = Subtree { delimiter: None, token_trees: attr.to_vec() };
                    let attr = ast::Attr::parse(&format!("#[{}]", tree)).ok()?;
                    // FIXME hygiene
                    let hygiene = Hygiene::new_unhygienic();
                    Attr::from_src(attr, &hygiene, index)
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
            let attrs = RawAttrs::from_attrs_owner(db, src.with_value(var as &dyn ast::AttrsOwner))
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
            let owner: &dyn AttrsOwner = match fld {
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

    pub fn docs(&self) -> Option<Documentation> {
        let docs = self.by_key("doc").attrs().flat_map(|attr| match attr.input.as_ref()? {
            AttrInput::Literal(s) => Some(s),
            AttrInput::TokenTree(_) => None,
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
                            it.as_ref().map(|it| it as &dyn ast::AttrsOwner),
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
                            ModuleSource::SourceFile(file) => file as &dyn ast::AttrsOwner,
                            ModuleSource::Module(module) => module as &dyn ast::AttrsOwner,
                            ModuleSource::BlockExpr(block) => block as &dyn ast::AttrsOwner,
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
                .left()
                .map_or_else(Default::default, |ast_id| attrs_from_ast(ast_id, db)),
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
                let attrs = match mod_data.declaration_source(db) {
                    Some(it) => {
                        let mut attrs: Vec<_> = collect_attrs(&it.value as &dyn ast::AttrsOwner)
                            .map(|attr| InFile::new(it.file_id, attr))
                            .collect();
                        if let InFile { file_id, value: ModuleSource::SourceFile(file) } =
                            mod_data.definition_source(db)
                        {
                            attrs.extend(
                                collect_attrs(&file as &dyn ast::AttrsOwner)
                                    .map(|attr| InFile::new(file_id, attr)),
                            )
                        }
                        attrs
                    }
                    None => {
                        let InFile { file_id, value } = mod_data.definition_source(db);
                        match &value {
                            ModuleSource::SourceFile(file) => {
                                collect_attrs(file as &dyn ast::AttrsOwner)
                            }
                            ModuleSource::Module(module) => {
                                collect_attrs(module as &dyn ast::AttrsOwner)
                            }
                            ModuleSource::BlockExpr(block) => {
                                collect_attrs(block as &dyn ast::AttrsOwner)
                            }
                        }
                        .map(|attr| InFile::new(file_id, attr))
                        .collect()
                    }
                };
                return AttrSourceMap { attrs };
            }
            AttrDefId::FieldId(id) => {
                id.parent.child_source(db).map(|source| match &source[id.local_id] {
                    Either::Left(field) => ast::AttrsOwnerNode::new(field.clone()),
                    Either::Right(field) => ast::AttrsOwnerNode::new(field.clone()),
                })
            }
            AttrDefId::AdtId(adt) => match adt {
                AdtId::StructId(id) => id.lookup(db).source(db).map(ast::AttrsOwnerNode::new),
                AdtId::UnionId(id) => id.lookup(db).source(db).map(ast::AttrsOwnerNode::new),
                AdtId::EnumId(id) => id.lookup(db).source(db).map(ast::AttrsOwnerNode::new),
            },
            AttrDefId::FunctionId(id) => id.lookup(db).source(db).map(ast::AttrsOwnerNode::new),
            AttrDefId::EnumVariantId(id) => id
                .parent
                .child_source(db)
                .map(|source| ast::AttrsOwnerNode::new(source[id.local_id].clone())),
            AttrDefId::StaticId(id) => id.lookup(db).source(db).map(ast::AttrsOwnerNode::new),
            AttrDefId::ConstId(id) => id.lookup(db).source(db).map(ast::AttrsOwnerNode::new),
            AttrDefId::TraitId(id) => id.lookup(db).source(db).map(ast::AttrsOwnerNode::new),
            AttrDefId::TypeAliasId(id) => id.lookup(db).source(db).map(ast::AttrsOwnerNode::new),
            AttrDefId::MacroDefId(id) => match id.ast_id() {
                Either::Left(it) => {
                    it.with_value(ast::AttrsOwnerNode::new(it.to_node(db.upcast())))
                }
                Either::Right(it) => {
                    it.with_value(ast::AttrsOwnerNode::new(it.to_node(db.upcast())))
                }
            },
            AttrDefId::ImplId(id) => id.lookup(db).source(db).map(ast::AttrsOwnerNode::new),
            AttrDefId::GenericParamId(id) => match id {
                GenericParamId::TypeParamId(id) => {
                    id.parent.child_source(db).map(|source| match &source[id.local_id] {
                        Either::Left(id) => ast::AttrsOwnerNode::new(id.clone()),
                        Either::Right(id) => ast::AttrsOwnerNode::new(id.clone()),
                    })
                }
                GenericParamId::LifetimeParamId(id) => id
                    .parent
                    .child_source(db)
                    .map(|source| ast::AttrsOwnerNode::new(source[id.local_id].clone())),
                GenericParamId::ConstParamId(id) => id
                    .parent
                    .child_source(db)
                    .map(|source| ast::AttrsOwnerNode::new(source[id.local_id].clone())),
            },
        };

        AttrSourceMap {
            attrs: collect_attrs(&owner.value)
                .map(|attr| InFile::new(owner.file_id, attr))
                .collect(),
        }
    }

    pub fn docs_with_rangemap(
        &self,
        db: &dyn DefDatabase,
    ) -> Option<(Documentation, DocsRangeMap)> {
        // FIXME: code duplication in `docs` above
        let docs = self.by_key("doc").attrs().flat_map(|attr| match attr.input.as_ref()? {
            AttrInput::Literal(s) => Some((s, attr.index)),
            AttrInput::TokenTree(_) => None,
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
            // str::lines doesn't yield anything for the empty string
            if !doc.is_empty() {
                for line in doc.split('\n') {
                    let line = line.trim_end();
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
                        TextRange::new(offset.try_into().ok()?, line_len.try_into().ok()?),
                    ));
                    buf.push('\n');
                }
            } else {
                buf.push('\n');
            }
        }
        buf.pop();
        if buf.is_empty() {
            None
        } else {
            Some((Documentation(buf), DocsRangeMap { mapping, source: self.source_map(db).attrs }))
        }
    }
}

fn inner_attributes(
    syntax: &SyntaxNode,
) -> Option<(impl Iterator<Item = ast::Attr>, impl Iterator<Item = ast::Comment>)> {
    let (attrs, docs) = match_ast! {
        match syntax {
            ast::SourceFile(it) => (it.attrs(), ast::CommentIter::from_syntax_node(it.syntax())),
            ast::ExternBlock(it) => {
                let extern_item_list = it.extern_item_list()?;
                (extern_item_list.attrs(), ast::CommentIter::from_syntax_node(extern_item_list.syntax()))
            },
            ast::Fn(it) => {
                let body = it.body()?;
                (body.attrs(), ast::CommentIter::from_syntax_node(body.syntax()))
            },
            ast::Impl(it) => {
                let assoc_item_list = it.assoc_item_list()?;
                (assoc_item_list.attrs(), ast::CommentIter::from_syntax_node(assoc_item_list.syntax()))
            },
            ast::Module(it) => {
                let item_list = it.item_list()?;
                (item_list.attrs(), ast::CommentIter::from_syntax_node(item_list.syntax()))
            },
            // FIXME: BlockExpr's only accept inner attributes in specific cases
            // Excerpt from the reference:
            // Block expressions accept outer and inner attributes, but only when they are the outer
            // expression of an expression statement or the final expression of another block expression.
            ast::BlockExpr(_it) => return None,
            _ => return None,
        }
    };
    let attrs = attrs.filter(|attr| attr.excl_token().is_some());
    let docs = docs.filter(|doc| doc.is_inner());
    Some((attrs, docs))
}

pub struct AttrSourceMap {
    attrs: Vec<InFile<Either<ast::Attr, ast::Comment>>>,
}

impl AttrSourceMap {
    /// Maps the lowered `Attr` back to its original syntax node.
    ///
    /// `attr` must come from the `owner` used for AttrSourceMap
    ///
    /// Note that the returned syntax node might be a `#[cfg_attr]`, or a doc comment, instead of
    /// the attribute represented by `Attr`.
    pub fn source_of(&self, attr: &Attr) -> InFile<&Either<ast::Attr, ast::Comment>> {
        self.attrs
            .get(attr.index as usize)
            .unwrap_or_else(|| panic!("cannot find `Attr` at index {}", attr.index))
            .as_ref()
    }
}

/// A struct to map text ranges from [`Documentation`] back to TextRanges in the syntax tree.
pub struct DocsRangeMap {
    source: Vec<InFile<Either<ast::Attr, ast::Comment>>>,
    // (docstring-line-range, attr_index, attr-string-range)
    // a mapping from the text range of a line of the [`Documentation`] to the attribute index and
    // the original (untrimmed) syntax doc line
    mapping: Vec<(TextRange, u32, TextRange)>,
}

impl DocsRangeMap {
    pub fn map(&self, range: TextRange) -> Option<InFile<TextRange>> {
        let found = self.mapping.binary_search_by(|(probe, ..)| probe.ordering(range)).ok()?;
        let (line_docs_range, idx, original_line_src_range) = self.mapping[found].clone();
        if !line_docs_range.contains_range(range) {
            return None;
        }

        let relative_range = range - line_docs_range.start();

        let &InFile { file_id, value: ref source } = &self.source[idx as usize];
        match source {
            Either::Left(_) => None, // FIXME, figure out a nice way to handle doc attributes here
            // as well as for whats done in syntax highlight doc injection
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attr {
    index: u32,
    pub(crate) path: Interned<ModPath>,
    pub(crate) input: Option<AttrInput>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttrInput {
    /// `#[attr = "string"]`
    Literal(SmolStr),
    /// `#[attr(subtree)]`
    TokenTree(Subtree),
}

impl Attr {
    fn from_src(ast: ast::Attr, hygiene: &Hygiene, index: u32) -> Option<Attr> {
        let path = Interned::new(ModPath::from_src(ast.path()?, hygiene)?);
        let input = if let Some(ast::Expr::Literal(lit)) = ast.expr() {
            let value = match lit.kind() {
                ast::LiteralKind::String(string) => string.value()?.into(),
                _ => lit.syntax().first_token()?.text().trim_matches('"').into(),
            };
            Some(AttrInput::Literal(value))
        } else if let Some(tt) = ast.token_tree() {
            Some(AttrInput::TokenTree(ast_to_token_tree(&tt).0))
        } else {
            None
        };
        Some(Attr { index, path, input })
    }

    /// Parses this attribute as a `#[derive]`, returns an iterator that yields all contained paths
    /// to derive macros.
    ///
    /// Returns `None` when the attribute is not a well-formed `#[derive]` attribute.
    pub(crate) fn parse_derive(&self) -> Option<impl Iterator<Item = ModPath>> {
        if self.path.as_ident() != Some(&hir_expand::name![derive]) {
            return None;
        }

        match &self.input {
            Some(AttrInput::TokenTree(args)) => {
                let mut counter = 0;
                let paths = args
                    .token_trees
                    .iter()
                    .group_by(move |tt| {
                        match tt {
                            tt::TokenTree::Leaf(tt::Leaf::Punct(p)) if p.char == ',' => {
                                counter += 1;
                            }
                            _ => {}
                        }
                        counter
                    })
                    .into_iter()
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
            _ => None,
        }
    }

    pub fn string_value(&self) -> Option<&SmolStr> {
        match self.input.as_ref()? {
            AttrInput::Literal(it) => Some(it),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AttrQuery<'a> {
    attrs: &'a Attrs,
    key: &'static str,
}

impl<'a> AttrQuery<'a> {
    pub fn tt_values(self) -> impl Iterator<Item = &'a Subtree> {
        self.attrs().filter_map(|attr| match attr.input.as_ref()? {
            AttrInput::TokenTree(it) => Some(it),
            _ => None,
        })
    }

    pub fn string_value(self) -> Option<&'a SmolStr> {
        self.attrs().find_map(|attr| match attr.input.as_ref()? {
            AttrInput::Literal(it) => Some(it),
            _ => None,
        })
    }

    pub fn exists(self) -> bool {
        self.attrs().next().is_some()
    }

    pub fn attrs(self) -> impl Iterator<Item = &'a Attr> + Clone {
        let key = self.key;
        self.attrs
            .iter()
            .filter(move |attr| attr.path.as_ident().map_or(false, |s| s.to_string() == key))
    }
}

fn attrs_from_ast<N>(src: AstId<N>, db: &dyn DefDatabase) -> RawAttrs
where
    N: ast::AttrsOwner,
{
    let src = InFile::new(src.file_id, src.to_node(db.upcast()));
    RawAttrs::from_attrs_owner(db, src.as_ref().map(|it| it as &dyn ast::AttrsOwner))
}

fn attrs_from_item_tree<N: ItemTreeNode>(id: ItemTreeId<N>, db: &dyn DefDatabase) -> RawAttrs {
    let tree = id.item_tree(db);
    let mod_item = N::id_to_mod_item(id.value);
    tree.raw_attrs(mod_item.into()).clone()
}

fn collect_attrs(
    owner: &dyn ast::AttrsOwner,
) -> impl Iterator<Item = Either<ast::Attr, ast::Comment>> {
    let (inner_attrs, inner_docs) = inner_attributes(owner.syntax())
        .map_or((None, None), |(attrs, docs)| (Some(attrs), Some(docs)));

    let outer_attrs = owner.attrs().filter(|attr| attr.excl_token().is_none());
    let attrs = outer_attrs
        .chain(inner_attrs.into_iter().flatten())
        .map(|attr| (attr.syntax().text_range().start(), Either::Left(attr)));

    let outer_docs =
        ast::CommentIter::from_syntax_node(owner.syntax()).filter(ast::Comment::is_outer);
    let docs = outer_docs
        .chain(inner_docs.into_iter().flatten())
        .map(|docs_text| (docs_text.syntax().text_range().start(), Either::Right(docs_text)));
    // sort here by syntax node offset because the source can have doc attributes and doc strings be interleaved
    let attrs: Vec<_> = docs.chain(attrs).sorted_by_key(|&(offset, _)| offset).collect();

    attrs.into_iter().map(|(_, attr)| attr)
}
