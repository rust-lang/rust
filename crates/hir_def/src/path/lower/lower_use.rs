//! Lowers a single complex use like `use foo::{bar, baz};` into a list of paths like
//! `foo::bar`, `foo::baz`;

use std::iter;

use either::Either;
use hir_expand::{hygiene::Hygiene, name::AsName};
use syntax::ast::{self, NameOwner};
use test_utils::mark;

use crate::path::{ImportAlias, ModPath, PathKind};

pub(crate) fn lower_use_tree(
    prefix: Option<ModPath>,
    tree: ast::UseTree,
    hygiene: &Hygiene,
    cb: &mut dyn FnMut(ModPath, &ast::UseTree, bool, Option<ImportAlias>),
) {
    if let Some(use_tree_list) = tree.use_tree_list() {
        let prefix = match tree.path() {
            // E.g. use something::{{{inner}}};
            None => prefix,
            // E.g. `use something::{inner}` (prefix is `None`, path is `something`)
            // or `use something::{path::{inner::{innerer}}}` (prefix is `something::path`, path is `inner`)
            Some(path) => match convert_path(prefix, path, hygiene) {
                Some(it) => Some(it),
                None => return, // FIXME: report errors somewhere
            },
        };
        for child_tree in use_tree_list.use_trees() {
            lower_use_tree(prefix.clone(), child_tree, hygiene, cb);
        }
    } else {
        let alias = tree.rename().map(|a| {
            a.name().map(|it| it.as_name()).map_or(ImportAlias::Underscore, ImportAlias::Alias)
        });
        let is_glob = tree.star_token().is_some();
        if let Some(ast_path) = tree.path() {
            // Handle self in a path.
            // E.g. `use something::{self, <...>}`
            if ast_path.qualifier().is_none() {
                if let Some(segment) = ast_path.segment() {
                    if segment.kind() == Some(ast::PathSegmentKind::SelfKw) {
                        if let Some(prefix) = prefix {
                            cb(prefix, &tree, false, alias);
                            return;
                        }
                    }
                }
            }
            if let Some(path) = convert_path(prefix, ast_path, hygiene) {
                cb(path, &tree, is_glob, alias)
            }
        // FIXME: report errors somewhere
        // We get here if we do
        } else if is_glob {
            mark::hit!(glob_enum_group);
            if let Some(prefix) = prefix {
                cb(prefix, &tree, is_glob, None)
            }
        }
    }
}

fn convert_path(prefix: Option<ModPath>, path: ast::Path, hygiene: &Hygiene) -> Option<ModPath> {
    let prefix = if let Some(qual) = path.qualifier() {
        Some(convert_path(prefix, qual, hygiene)?)
    } else {
        prefix
    };

    let segment = path.segment()?;
    let res = match segment.kind()? {
        ast::PathSegmentKind::Name(name_ref) => {
            match hygiene.name_ref_to_name(name_ref) {
                Either::Left(name) => {
                    // no type args in use
                    let mut res = prefix.unwrap_or_else(|| ModPath {
                        kind: segment.coloncolon_token().map_or(PathKind::Plain, |_| PathKind::Abs),
                        segments: Vec::with_capacity(1),
                    });
                    res.segments.push(name);
                    res
                }
                Either::Right(crate_id) => {
                    return Some(ModPath::from_segments(
                        PathKind::DollarCrate(crate_id),
                        iter::empty(),
                    ))
                }
            }
        }
        ast::PathSegmentKind::CrateKw => {
            if prefix.is_some() {
                return None;
            }
            ModPath::from_segments(PathKind::Crate, iter::empty())
        }
        ast::PathSegmentKind::SelfKw => {
            if prefix.is_some() {
                return None;
            }
            ModPath::from_segments(PathKind::Super(0), iter::empty())
        }
        ast::PathSegmentKind::SuperKw => {
            let nested_super_count = match prefix.map(|p| p.kind) {
                Some(PathKind::Super(n)) => n,
                Some(_) => return None,
                None => 0,
            };

            ModPath::from_segments(PathKind::Super(nested_super_count + 1), iter::empty())
        }
        ast::PathSegmentKind::Type { .. } => {
            // not allowed in imports
            return None;
        }
    };
    Some(res)
}
