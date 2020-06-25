//! A higher level attributes based on TokenTree, with also some shortcuts.

use std::{ops, sync::Arc};

use either::Either;
use hir_expand::{hygiene::Hygiene, AstId, InFile};
use mbe::ast_to_token_tree;
use ra_cfg::CfgOptions;
use ra_syntax::{
    ast::{self, AstNode, AttrsOwner},
    SmolStr,
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

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct Attrs {
    entries: Option<Arc<[Attr]>>,
}

impl ops::Deref for Attrs {
    type Target = [Attr];

    fn deref(&self) -> &[Attr] {
        match &self.entries {
            Some(it) => &*it,
            None => &[],
        }
    }
}

impl Attrs {
    pub const EMPTY: Attrs = Attrs { entries: None };

    pub(crate) fn attrs_query(db: &dyn DefDatabase, def: AttrDefId) -> Attrs {
        match def {
            AttrDefId::ModuleId(module) => {
                let def_map = db.crate_def_map(module.krate);
                let mod_data = &def_map[module.local_id];
                match mod_data.declaration_source(db) {
                    Some(it) => {
                        Attrs::from_attrs_owner(db, it.as_ref().map(|it| it as &dyn AttrsOwner))
                    }
                    None => Attrs::from_attrs_owner(
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
                    Either::Left(_tuple) => Attrs::default(),
                    Either::Right(record) => Attrs::from_attrs_owner(db, src.with_value(record)),
                }
            }
            AttrDefId::EnumVariantId(var_id) => {
                let src = var_id.parent.child_source(db);
                let src = src.as_ref().map(|it| &it[var_id.local_id]);
                Attrs::from_attrs_owner(db, src.map(|it| it as &dyn AttrsOwner))
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
        }
    }

    pub fn from_attrs_owner(db: &dyn DefDatabase, owner: InFile<&dyn AttrsOwner>) -> Attrs {
        let hygiene = Hygiene::new(db.upcast(), owner.file_id);
        Attrs::new(owner.value, &hygiene)
    }

    pub(crate) fn new(owner: &dyn AttrsOwner, hygiene: &Hygiene) -> Attrs {
        let docs = ast::CommentIter::from_syntax_node(owner.syntax()).doc_comment_text().map(
            |docs_text| Attr {
                input: Some(AttrInput::Literal(SmolStr::new(docs_text))),
                path: ModPath::from(hir_expand::name!(doc)),
            },
        );
        let mut attrs = owner.attrs().peekable();
        let entries = if attrs.peek().is_none() {
            // Avoid heap allocation
            None
        } else {
            Some(attrs.flat_map(|ast| Attr::from_src(ast, hygiene)).chain(docs).collect())
        };
        Attrs { entries }
    }

    pub fn merge(&self, other: Attrs) -> Attrs {
        match (&self.entries, &other.entries) {
            (None, None) => Attrs { entries: None },
            (Some(entries), None) | (None, Some(entries)) => {
                Attrs { entries: Some(entries.clone()) }
            }
            (Some(a), Some(b)) => {
                Attrs { entries: Some(a.iter().chain(b.iter()).cloned().collect()) }
            }
        }
    }

    pub fn by_key(&self, key: &'static str) -> AttrQuery<'_> {
        AttrQuery { attrs: self, key }
    }

    pub(crate) fn is_cfg_enabled(&self, cfg_options: &CfgOptions) -> bool {
        // FIXME: handle cfg_attr :-)
        self.by_key("cfg").tt_values().all(|tt| cfg_options.is_cfg_enabled(tt) != Some(false))
    }
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
        let input = match ast.input() {
            None => None,
            Some(ast::AttrInput::Literal(lit)) => {
                // FIXME: escape? raw string?
                let value = lit.syntax().first_token()?.text().trim_matches('"').into();
                Some(AttrInput::Literal(value))
            }
            Some(ast::AttrInput::TokenTree(tt)) => {
                Some(AttrInput::TokenTree(ast_to_token_tree(&tt)?.0))
            }
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

fn attrs_from_ast<N>(src: AstId<N>, db: &dyn DefDatabase) -> Attrs
where
    N: ast::AttrsOwner,
{
    let src = InFile::new(src.file_id, src.to_node(db.upcast()));
    Attrs::from_attrs_owner(db, src.as_ref().map(|it| it as &dyn AttrsOwner))
}

fn attrs_from_item_tree<N: ItemTreeNode>(id: ItemTreeId<N>, db: &dyn DefDatabase) -> Attrs {
    let tree = db.item_tree(id.file_id);
    let mod_item = N::id_to_mod_item(id.value);
    tree.attrs(mod_item.into()).clone()
}
