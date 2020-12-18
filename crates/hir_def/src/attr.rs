//! A higher level attributes based on TokenTree, with also some shortcuts.

use std::{ops, sync::Arc};

use base_db::CrateId;
use cfg::{CfgExpr, CfgOptions};
use either::Either;
use hir_expand::{hygiene::Hygiene, AstId, InFile};
use itertools::Itertools;
use mbe::ast_to_token_tree;
use syntax::{
    ast::{self, AstNode, AttrsOwner},
    match_ast, AstToken, SmolStr, SyntaxNode,
};
use tt::Subtree;

use crate::{
    db::DefDatabase,
    item_tree::{ItemTreeId, ItemTreeNode},
    nameres::ModuleSource,
    path::ModPath,
    src::HasChildSource,
    AdtId, AttrDefId, Lookup,
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
pub struct RawAttrs {
    entries: Option<Arc<[Attr]>>,
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct Attrs(RawAttrs);

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

impl RawAttrs {
    pub const EMPTY: Self = Self { entries: None };

    pub(crate) fn new(owner: &dyn AttrsOwner, hygiene: &Hygiene) -> Self {
        let (inner_attrs, inner_docs) = inner_attributes(owner.syntax())
            .map_or((None, None), |(attrs, docs)| ((Some(attrs), Some(docs))));

        let outer_attrs = owner.attrs().filter(|attr| attr.excl_token().is_none());
        let attrs = outer_attrs
            .chain(inner_attrs.into_iter().flatten())
            .map(|attr| (attr.syntax().text_range().start(), Attr::from_src(attr, hygiene)));

        let outer_docs =
            ast::CommentIter::from_syntax_node(owner.syntax()).filter(ast::Comment::is_outer);
        let docs = outer_docs.chain(inner_docs.into_iter().flatten()).map(|docs_text| {
            (
                docs_text.syntax().text_range().start(),
                docs_text.doc_comment().map(|doc| Attr {
                    input: Some(AttrInput::Literal(SmolStr::new(doc))),
                    path: ModPath::from(hir_expand::name!(doc)),
                }),
            )
        });
        // sort here by syntax node offset because the source can have doc attributes and doc strings be interleaved
        let attrs: Vec<_> = docs.chain(attrs).sorted_by_key(|&(offset, _)| offset).collect();
        let entries = if attrs.is_empty() {
            // Avoid heap allocation
            None
        } else {
            Some(attrs.into_iter().flat_map(|(_, attr)| attr).collect())
        };
        Self { entries }
    }

    fn from_attrs_owner(db: &dyn DefDatabase, owner: InFile<&dyn AttrsOwner>) -> Self {
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
            .filter_map(|attr| {
                let attr = attr.clone();
                let is_cfg_attr =
                    attr.path.as_ident().map_or(false, |name| *name == hir_expand::name![cfg_attr]);
                if !is_cfg_attr {
                    return Some(attr);
                }

                let subtree = match &attr.input {
                    Some(AttrInput::TokenTree(it)) => it,
                    _ => return Some(attr),
                };

                // Input subtree is: `(cfg, attr)`
                // Split it up into a `cfg` and an `attr` subtree.
                // FIXME: There should be a common API for this.
                let mut saw_comma = false;
                let (mut cfg, attr): (Vec<_>, Vec<_>) =
                    subtree.clone().token_trees.into_iter().partition(|tree| {
                        if saw_comma {
                            return false;
                        }

                        match tree {
                            tt::TokenTree::Leaf(tt::Leaf::Punct(p)) if p.char == ',' => {
                                saw_comma = true;
                            }
                            _ => {}
                        }

                        true
                    });
                cfg.pop(); // `,` ends up in here

                let cfg = Subtree { delimiter: subtree.delimiter, token_trees: cfg };
                let cfg = CfgExpr::parse(&cfg);

                let cfg_options = &crate_graph[krate].cfg_options;
                if cfg_options.check(&cfg) == Some(false) {
                    None
                } else {
                    let attr = Subtree { delimiter: None, token_trees: attr };
                    let attr = ast::Attr::parse(&attr.to_string()).ok()?;
                    let hygiene = Hygiene::new_unhygienic(); // FIXME
                    Attr::from_src(attr, &hygiene)
                }
            })
            .collect();

        Attrs(RawAttrs { entries: Some(new_attrs) })
    }
}

impl Attrs {
    pub const EMPTY: Self = Self(RawAttrs::EMPTY);

    pub(crate) fn attrs_query(db: &dyn DefDatabase, def: AttrDefId) -> Attrs {
        let raw_attrs = match def {
            AttrDefId::ModuleId(module) => {
                let def_map = db.crate_def_map(module.krate);
                let mod_data = &def_map[module.local_id];
                match mod_data.declaration_source(db) {
                    Some(it) => {
                        RawAttrs::from_attrs_owner(db, it.as_ref().map(|it| it as &dyn AttrsOwner))
                    }
                    None => RawAttrs::from_attrs_owner(
                        db,
                        mod_data.definition_source(db).as_ref().map(|src| match src {
                            ModuleSource::SourceFile(file) => file as &dyn AttrsOwner,
                            ModuleSource::Module(module) => module as &dyn AttrsOwner,
                        }),
                    ),
                }
            }
            AttrDefId::FieldId(it) => {
                let src = it.parent.child_source(db);
                match &src.value[it.local_id] {
                    Either::Left(_tuple) => RawAttrs::default(),
                    Either::Right(record) => RawAttrs::from_attrs_owner(db, src.with_value(record)),
                }
            }
            AttrDefId::EnumVariantId(var_id) => {
                let src = var_id.parent.child_source(db);
                let src = src.as_ref().map(|it| &it[var_id.local_id]);
                RawAttrs::from_attrs_owner(db, src.map(|it| it as &dyn AttrsOwner))
            }
            AttrDefId::AdtId(it) => match it {
                AdtId::StructId(it) => attrs_from_item_tree(it.lookup(db).id, db),
                AdtId::EnumId(it) => attrs_from_item_tree(it.lookup(db).id, db),
                AdtId::UnionId(it) => attrs_from_item_tree(it.lookup(db).id, db),
            },
            AttrDefId::TraitId(it) => attrs_from_item_tree(it.lookup(db).id, db),
            AttrDefId::MacroDefId(it) => {
                it.ast_id.map_or_else(Default::default, |ast_id| attrs_from_ast(ast_id, db))
            }
            AttrDefId::ImplId(it) => attrs_from_item_tree(it.lookup(db).id, db),
            AttrDefId::ConstId(it) => attrs_from_item_tree(it.lookup(db).id, db),
            AttrDefId::StaticId(it) => attrs_from_item_tree(it.lookup(db).id, db),
            AttrDefId::FunctionId(it) => attrs_from_item_tree(it.lookup(db).id, db),
            AttrDefId::TypeAliasId(it) => attrs_from_item_tree(it.lookup(db).id, db),
        };

        raw_attrs.filter(db, def.krate(db))
    }

    pub fn merge(&self, other: Attrs) -> Attrs {
        match (&self.0.entries, &other.0.entries) {
            (None, None) => Attrs::EMPTY,
            (Some(entries), None) | (None, Some(entries)) => {
                Attrs(RawAttrs { entries: Some(entries.clone()) })
            }
            (Some(a), Some(b)) => {
                Attrs(RawAttrs { entries: Some(a.iter().chain(b.iter()).cloned().collect()) })
            }
        }
    }

    pub fn by_key(&self, key: &'static str) -> AttrQuery<'_> {
        AttrQuery { attrs: self, key }
    }

    pub fn cfg(&self) -> Option<CfgExpr> {
        // FIXME: handle cfg_attr :-)
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
        let docs = self
            .by_key("doc")
            .attrs()
            .flat_map(|attr| match attr.input.as_ref()? {
                AttrInput::Literal(s) => Some(s),
                AttrInput::TokenTree(_) => None,
            })
            .intersperse(&SmolStr::new_inline("\n"))
            .map(|it| it.as_str())
            .collect::<String>();
        if docs.is_empty() {
            None
        } else {
            Some(Documentation(docs.into()))
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
            ast::BlockExpr(it) => return None,
            _ => return None,
        }
    };
    let attrs = attrs.filter(|attr| attr.excl_token().is_some());
    let docs = docs.filter(|doc| doc.is_inner());
    Some((attrs, docs))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attr {
    pub(crate) path: ModPath,
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
    fn from_src(ast: ast::Attr, hygiene: &Hygiene) -> Option<Attr> {
        let path = ModPath::from_src(ast.path()?, hygiene)?;
        let input = if let Some(lit) = ast.literal() {
            let value = match lit.kind() {
                ast::LiteralKind::String(string) => string.value()?.into(),
                _ => lit.syntax().first_token()?.text().trim_matches('"').into(),
            };
            Some(AttrInput::Literal(value))
        } else if let Some(tt) = ast.token_tree() {
            Some(AttrInput::TokenTree(ast_to_token_tree(&tt)?.0))
        } else {
            None
        };
        Some(Attr { path, input })
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

    fn attrs(self) -> impl Iterator<Item = &'a Attr> {
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
    RawAttrs::from_attrs_owner(db, src.as_ref().map(|it| it as &dyn AttrsOwner))
}

fn attrs_from_item_tree<N: ItemTreeNode>(id: ItemTreeId<N>, db: &dyn DefDatabase) -> RawAttrs {
    let tree = db.item_tree(id.file_id);
    let mod_item = N::id_to_mod_item(id.value);
    tree.raw_attrs(mod_item.into()).clone()
}
