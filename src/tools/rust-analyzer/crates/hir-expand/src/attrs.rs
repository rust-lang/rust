//! A higher level attributes based on TokenTree, with also some shortcuts.
use std::{fmt, ops, sync::Arc};

use base_db::CrateId;
use cfg::CfgExpr;
use either::Either;
use intern::Interned;
use mbe::{syntax_node_to_token_tree, DelimiterKind, Punct};
use smallvec::{smallvec, SmallVec};
use syntax::{ast, match_ast, AstNode, SmolStr, SyntaxNode};

use crate::{
    db::AstDatabase,
    hygiene::Hygiene,
    mod_path::{ModPath, PathKind},
    name::AsName,
    tt::{self, Subtree},
    InFile,
};

/// Syntactical attributes, without filtering of `cfg_attr`s.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct RawAttrs {
    entries: Option<Arc<[Attr]>>,
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

impl RawAttrs {
    pub const EMPTY: Self = Self { entries: None };

    pub fn new(db: &dyn AstDatabase, owner: &dyn ast::HasAttrs, hygiene: &Hygiene) -> Self {
        let entries = collect_attrs(owner)
            .filter_map(|(id, attr)| match attr {
                Either::Left(attr) => {
                    attr.meta().and_then(|meta| Attr::from_src(db, meta, hygiene, id))
                }
                Either::Right(comment) => comment.doc_comment().map(|doc| Attr {
                    id,
                    input: Some(Interned::new(AttrInput::Literal(SmolStr::new(doc)))),
                    path: Interned::new(ModPath::from(crate::name!(doc))),
                }),
            })
            .collect::<Arc<_>>();

        Self { entries: if entries.is_empty() { None } else { Some(entries) } }
    }

    pub fn from_attrs_owner(db: &dyn AstDatabase, owner: InFile<&dyn ast::HasAttrs>) -> Self {
        let hygiene = Hygiene::new(db, owner.file_id);
        Self::new(db, owner.value, &hygiene)
    }

    pub fn merge(&self, other: Self) -> Self {
        match (&self.entries, other.entries) {
            (None, None) => Self::EMPTY,
            (None, entries @ Some(_)) => Self { entries },
            (Some(entries), None) => Self { entries: Some(entries.clone()) },
            (Some(a), Some(b)) => {
                let last_ast_index = a.last().map_or(0, |it| it.id.ast_index() + 1) as u32;
                Self {
                    entries: Some(
                        a.iter()
                            .cloned()
                            .chain(b.iter().map(|it| {
                                let mut it = it.clone();
                                it.id.id = it.id.ast_index() as u32 + last_ast_index
                                    | (it.id.cfg_attr_index().unwrap_or(0) as u32)
                                        << AttrId::AST_INDEX_BITS;
                                it
                            }))
                            .collect(),
                    ),
                }
            }
        }
    }

    /// Processes `cfg_attr`s, returning the resulting semantic `Attrs`.
    // FIXME: This should return a different type
    pub fn filter(self, db: &dyn AstDatabase, krate: CrateId) -> RawAttrs {
        let has_cfg_attrs = self
            .iter()
            .any(|attr| attr.path.as_ident().map_or(false, |name| *name == crate::name![cfg_attr]));
        if !has_cfg_attrs {
            return self;
        }

        let crate_graph = db.crate_graph();
        let new_attrs = self
            .iter()
            .flat_map(|attr| -> SmallVec<[_; 1]> {
                let is_cfg_attr =
                    attr.path.as_ident().map_or(false, |name| *name == crate::name![cfg_attr]);
                if !is_cfg_attr {
                    return smallvec![attr.clone()];
                }

                let subtree = match attr.token_tree_value() {
                    Some(it) => it,
                    _ => return smallvec![attr.clone()],
                };

                let (cfg, parts) = match parse_cfg_attr_input(subtree) {
                    Some(it) => it,
                    None => return smallvec![attr.clone()],
                };
                let index = attr.id;
                let attrs =
                    parts.enumerate().take(1 << AttrId::CFG_ATTR_BITS).filter_map(|(idx, attr)| {
                        let tree = Subtree {
                            delimiter: tt::Delimiter::unspecified(),
                            token_trees: attr.to_vec(),
                        };
                        // FIXME hygiene
                        let hygiene = Hygiene::new_unhygienic();
                        Attr::from_tt(db, &tree, &hygiene, index.with_cfg_attr(idx))
                    });

                let cfg_options = &crate_graph[krate].cfg_options;
                let cfg = Subtree { delimiter: subtree.delimiter, token_trees: cfg.to_vec() };
                let cfg = CfgExpr::parse(&cfg);
                if cfg_options.check(&cfg) == Some(false) {
                    smallvec![]
                } else {
                    cov_mark::hit!(cfg_attr_active);

                    attrs.collect()
                }
            })
            .collect();

        RawAttrs { entries: Some(new_attrs) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AttrId {
    id: u32,
}

// FIXME: This only handles a single level of cfg_attr nesting
// that is `#[cfg_attr(all(), cfg_attr(all(), cfg(any())))]` breaks again
impl AttrId {
    const CFG_ATTR_BITS: usize = 7;
    const AST_INDEX_MASK: usize = 0x00FF_FFFF;
    const AST_INDEX_BITS: usize = Self::AST_INDEX_MASK.count_ones() as usize;
    const CFG_ATTR_SET_BITS: u32 = 1 << 31;

    pub fn ast_index(&self) -> usize {
        self.id as usize & Self::AST_INDEX_MASK
    }

    pub fn cfg_attr_index(&self) -> Option<usize> {
        if self.id & Self::CFG_ATTR_SET_BITS == 0 {
            None
        } else {
            Some(self.id as usize >> Self::AST_INDEX_BITS)
        }
    }

    pub fn with_cfg_attr(self, idx: usize) -> AttrId {
        AttrId { id: self.id | (idx as u32) << Self::AST_INDEX_BITS | Self::CFG_ATTR_SET_BITS }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attr {
    pub id: AttrId,
    pub path: Interned<ModPath>,
    pub input: Option<Interned<AttrInput>>,
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
        db: &dyn AstDatabase,
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
        db: &dyn AstDatabase,
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

        if args.delimiter.kind != DelimiterKind::Parenthesis {
            return None;
        }
        let paths = args
            .token_trees
            .split(|tt| matches!(tt, tt::TokenTree::Leaf(tt::Leaf::Punct(Punct { char: ',', .. }))))
            .filter_map(|tts| {
                if tts.is_empty() {
                    return None;
                }
                let segments = tts.iter().filter_map(|tt| match tt {
                    tt::TokenTree::Leaf(tt::Leaf::Ident(id)) => Some(id.as_name()),
                    _ => None,
                });
                Some(ModPath::from_segments(PathKind::Plain, segments))
            });

        Some(paths)
    }
}

pub fn collect_attrs(
    owner: &dyn ast::HasAttrs,
) -> impl Iterator<Item = (AttrId, Either<ast::Attr, ast::Comment>)> {
    let inner_attrs = inner_attributes(owner.syntax()).into_iter().flatten();
    let outer_attrs =
        ast::AttrDocCommentIter::from_syntax_node(owner.syntax()).filter(|el| match el {
            Either::Left(attr) => attr.kind().is_outer(),
            Either::Right(comment) => comment.is_outer(),
        });
    outer_attrs.chain(inner_attrs).enumerate().map(|(id, attr)| (AttrId { id: id as u32 }, attr))
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

// Input subtree is: `(cfg, $(attr),+)`
// Split it up into a `cfg` subtree and the `attr` subtrees.
pub fn parse_cfg_attr_input(
    subtree: &Subtree,
) -> Option<(&[tt::TokenTree], impl Iterator<Item = &[tt::TokenTree]>)> {
    let mut parts = subtree
        .token_trees
        .split(|tt| matches!(tt, tt::TokenTree::Leaf(tt::Leaf::Punct(Punct { char: ',', .. }))));
    let cfg = parts.next()?;
    Some((cfg, parts.filter(|it| !it.is_empty())))
}
