use ra_syntax::{SmolStr, ast, AstNode};

use crate::syntax_ptr::LocalSyntaxPtr;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Path {
    pub(crate) kind: PathKind,
    pub(crate) segments: Vec<SmolStr>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PathKind {
    Abs,
    Self_,
    Super,
    Crate,
}

impl Path {
    pub(crate) fn expand_use_item(
        item: ast::UseItem,
        mut cb: impl FnMut(Path, Option<LocalSyntaxPtr>),
    ) {
        if let Some(tree) = item.use_tree() {
            expand_use_tree(None, tree, &mut cb);
        }
    }
}

fn expand_use_tree(
    prefix: Option<Path>,
    tree: ast::UseTree,
    cb: &mut impl FnMut(Path, Option<LocalSyntaxPtr>),
) {
    if let Some(use_tree_list) = tree.use_tree_list() {
        let prefix = match tree.path() {
            None => prefix,
            Some(path) => match convert_path(prefix, path) {
                Some(it) => Some(it),
                None => return, // TODO: report errors somewhere
            },
        };
        for tree in use_tree_list.use_trees() {
            expand_use_tree(prefix.clone(), tree, cb);
        }
    } else {
        if let Some(ast_path) = tree.path() {
            if let Some(path) = convert_path(prefix, ast_path) {
                let ptr = if tree.has_star() {
                    None
                } else {
                    let ptr = LocalSyntaxPtr::new(ast_path.segment().unwrap().syntax());
                    Some(ptr)
                };
                cb(path, ptr)
            }
        }
    }
}

fn convert_path(prefix: Option<Path>, path: ast::Path) -> Option<Path> {
    let prefix = if let Some(qual) = path.qualifier() {
        Some(convert_path(prefix, qual)?)
    } else {
        None
    };
    let segment = path.segment()?;
    let res = match segment.kind()? {
        ast::PathSegmentKind::Name(name) => {
            let mut res = prefix.unwrap_or_else(|| Path {
                kind: PathKind::Abs,
                segments: Vec::with_capacity(1),
            });
            res.segments.push(name.text());
            res
        }
        ast::PathSegmentKind::CrateKw => {
            if prefix.is_some() {
                return None;
            }
            Path {
                kind: PathKind::Crate,
                segments: Vec::new(),
            }
        }
        ast::PathSegmentKind::SelfKw => {
            if prefix.is_some() {
                return None;
            }
            Path {
                kind: PathKind::Self_,
                segments: Vec::new(),
            }
        }
        ast::PathSegmentKind::SuperKw => {
            if prefix.is_some() {
                return None;
            }
            Path {
                kind: PathKind::Super,
                segments: Vec::new(),
            }
        }
    };
    Some(res)
}
