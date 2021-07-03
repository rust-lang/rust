use hir::AsAssocItem;
use ide_db::helpers::{
    insert_use::{insert_use, ImportScope},
    mod_path_to_ast,
};
use syntax::{ast, match_ast, ted, AstNode, SyntaxNode};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: replace_qualified_name_with_use
//
// Adds a use statement for a given fully-qualified name.
//
// ```
// # mod std { pub mod collections { pub struct HashMap<T, U>(T, U); } }
// fn process(map: std::collections::$0HashMap<String, String>) {}
// ```
// ->
// ```
// use std::collections::HashMap;
//
// # mod std { pub mod collections { pub struct HashMap<T, U>(T, U); } }
// fn process(map: HashMap<String, String>) {}
// ```
pub(crate) fn replace_qualified_name_with_use(
    acc: &mut Assists,
    ctx: &AssistContext,
) -> Option<()> {
    let path: ast::Path = ctx.find_node_at_offset()?;
    // We don't want to mess with use statements
    if path.syntax().ancestors().find_map(ast::UseTree::cast).is_some() {
        cov_mark::hit!(not_applicable_in_use);
        return None;
    }

    if path.qualifier().is_none() {
        cov_mark::hit!(dont_import_trivial_paths);
        return None;
    }

    let res = ctx.sema.resolve_path(&path)?;
    let def: hir::ItemInNs = match res {
        hir::PathResolution::Def(def) if def.as_assoc_item(ctx.sema.db).is_none() => def.into(),
        hir::PathResolution::Macro(mac) => mac.into(),
        _ => return None,
    };

    let target = path.syntax().text_range();
    let scope = ImportScope::find_insert_use_container_with_macros(path.syntax(), &ctx.sema)?;
    let mod_path = ctx.sema.scope(path.syntax()).module()?.find_use_path_prefixed(
        ctx.sema.db,
        def,
        ctx.config.insert_use.prefix_kind,
    )?;
    acc.add(
        AssistId("replace_qualified_name_with_use", AssistKind::RefactorRewrite),
        "Replace qualified path with use",
        target,
        |builder| {
            // Now that we've brought the name into scope, re-qualify all paths that could be
            // affected (that is, all paths inside the node we added the `use` to).
            let scope = match scope {
                ImportScope::File(it) => ImportScope::File(builder.make_mut(it)),
                ImportScope::Module(it) => ImportScope::Module(builder.make_mut(it)),
                ImportScope::Block(it) => ImportScope::Block(builder.make_mut(it)),
            };
            let path = mod_path_to_ast(&mod_path);
            shorten_paths(scope.as_syntax_node(), &path.clone_for_update());
            insert_use(&scope, path, &ctx.config.insert_use);
        },
    )
}

/// Adds replacements to `re` that shorten `path` in all descendants of `node`.
fn shorten_paths(node: &SyntaxNode, path: &ast::Path) {
    for child in node.children() {
        match_ast! {
            match child {
                // Don't modify `use` items, as this can break the `use` item when injecting a new
                // import into the use tree.
                ast::Use(_it) => continue,
                // Don't descend into submodules, they don't have the same `use` items in scope.
                // FIXME: This isn't true due to `super::*` imports?
                ast::Module(_it) => continue,
                ast::Path(p) => if maybe_replace_path(p.clone(), path.clone()).is_none() {
                    shorten_paths(p.syntax(), path);
                },
                _ => shorten_paths(&child, path),
            }
        }
    }
}

fn maybe_replace_path(path: ast::Path, target: ast::Path) -> Option<()> {
    if !path_eq_no_generics(path.clone(), target) {
        return None;
    }

    // Shorten `path`, leaving only its last segment.
    if let Some(parent) = path.qualifier() {
        ted::remove(parent.syntax());
    }
    if let Some(double_colon) = path.coloncolon_token() {
        ted::remove(&double_colon);
    }

    Some(())
}

fn path_eq_no_generics(lhs: ast::Path, rhs: ast::Path) -> bool {
    let mut lhs_curr = lhs;
    let mut rhs_curr = rhs;
    loop {
        match lhs_curr.segment().zip(rhs_curr.segment()) {
            Some((lhs, rhs))
                if lhs.coloncolon_token().is_some() == rhs.coloncolon_token().is_some()
                    && lhs
                        .name_ref()
                        .zip(rhs.name_ref())
                        .map_or(false, |(lhs, rhs)| lhs.text() == rhs.text()) =>
            {
                ()
            }
            _ => return false,
        }

        match (lhs_curr.qualifier(), rhs_curr.qualifier()) {
            (Some(lhs), Some(rhs)) => {
                lhs_curr = lhs;
                rhs_curr = rhs;
            }
            (None, None) => return true,
            _ => return false,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn test_replace_already_imported() {
        check_assist(
            replace_qualified_name_with_use,
            r"
mod std { pub mod fs { pub struct Path; } }
use std::fs;

fn main() {
    std::f$0s::Path
}",
            r"
mod std { pub mod fs { pub struct Path; } }
use std::fs;

fn main() {
    fs::Path
}",
        )
    }

    #[test]
    fn test_replace_add_use_no_anchor() {
        check_assist(
            replace_qualified_name_with_use,
            r"
mod std { pub mod fs { pub struct Path; } }
std::fs::Path$0
    ",
            r"
use std::fs::Path;

mod std { pub mod fs { pub struct Path; } }
Path
    ",
        );
    }

    #[test]
    fn test_replace_add_use_no_anchor_middle_segment() {
        check_assist(
            replace_qualified_name_with_use,
            r"
mod std { pub mod fs { pub struct Path; } }
std::fs$0::Path
    ",
            r"
use std::fs;

mod std { pub mod fs { pub struct Path; } }
fs::Path
    ",
        );
    }
    #[test]
    #[test]
    fn dont_import_trivial_paths() {
        cov_mark::check!(dont_import_trivial_paths);
        check_assist_not_applicable(replace_qualified_name_with_use, r"impl foo$0 for () {}");
    }

    #[test]
    fn test_replace_not_applicable_in_use() {
        cov_mark::check!(not_applicable_in_use);
        check_assist_not_applicable(replace_qualified_name_with_use, r"use std::fmt$0;");
    }

    #[test]
    fn replaces_all_affected_paths() {
        check_assist(
            replace_qualified_name_with_use,
            r"
mod std { pub mod fmt { pub trait Debug {} } }
fn main() {
    std::fmt::Debug$0;
    let x: std::fmt::Debug = std::fmt::Debug;
}
    ",
            r"
use std::fmt::Debug;

mod std { pub mod fmt { pub trait Debug {} } }
fn main() {
    Debug;
    let x: Debug = Debug;
}
    ",
        );
    }

    #[test]
    fn does_not_replace_in_submodules() {
        check_assist(
            replace_qualified_name_with_use,
            r"
mod std { pub mod fmt { pub trait Debug {} } }
fn main() {
    std::fmt::Debug$0;
}

mod sub {
    fn f() {
        std::fmt::Debug;
    }
}
    ",
            r"
use std::fmt::Debug;

mod std { pub mod fmt { pub trait Debug {} } }
fn main() {
    Debug;
}

mod sub {
    fn f() {
        std::fmt::Debug;
    }
}
    ",
        );
    }

    #[test]
    fn does_not_replace_in_use() {
        check_assist(
            replace_qualified_name_with_use,
            r"
mod std { pub mod fmt { pub trait Display {} } }
use std::fmt::Display;

fn main() {
    std::fmt$0;
}
    ",
            r"
mod std { pub mod fmt { pub trait Display {} } }
use std::fmt::{self, Display};

fn main() {
    fmt;
}
    ",
        );
    }

    #[test]
    fn does_not_replace_assoc_item_path() {
        check_assist_not_applicable(
            replace_qualified_name_with_use,
            r"
pub struct Foo;
impl Foo {
    pub fn foo() {}
}

fn main() {
    Foo::foo$0();
}
",
        );
    }
}
