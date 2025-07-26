use hir::AsAssocItem;
use ide_db::{
    helpers::mod_path_to_ast,
    imports::insert_use::{ImportScope, insert_use},
};
use syntax::{
    AstNode, Edition, SyntaxNode,
    ast::{self, HasGenericArgs, make},
    match_ast, ted,
};

use crate::{AssistContext, AssistId, Assists};

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
    ctx: &AssistContext<'_>,
) -> Option<()> {
    let mut original_path: ast::Path = ctx.find_node_at_offset()?;
    // We don't want to mess with use statements
    if original_path.syntax().ancestors().find_map(ast::UseTree::cast).is_some() {
        cov_mark::hit!(not_applicable_in_use);
        return None;
    }

    if original_path.qualifier().is_none() {
        original_path = original_path.parent_path()?;
    }

    // only offer replacement for non assoc items
    match ctx.sema.resolve_path(&original_path)? {
        hir::PathResolution::Def(def) if def.as_assoc_item(ctx.sema.db).is_none() => (),
        _ => return None,
    }
    // then search for an import for the first path segment of what we want to replace
    // that way it is less likely that we import the item from a different location due re-exports
    let module = match ctx.sema.resolve_path(&original_path.first_qualifier_or_self())? {
        hir::PathResolution::Def(module @ hir::ModuleDef::Module(_)) => module,
        _ => return None,
    };

    let starts_with_name_ref = !matches!(
        original_path.first_segment().and_then(|it| it.kind()),
        Some(
            ast::PathSegmentKind::CrateKw
                | ast::PathSegmentKind::SuperKw
                | ast::PathSegmentKind::SelfKw
        )
    );
    let path_to_qualifier = starts_with_name_ref
        .then(|| {
            ctx.sema.scope(original_path.syntax())?.module().find_use_path(
                ctx.sema.db,
                module,
                ctx.config.insert_use.prefix_kind,
                ctx.config.import_path_config(),
            )
        })
        .flatten();

    let scope = ImportScope::find_insert_use_container(original_path.syntax(), &ctx.sema)?;
    let target = original_path.syntax().text_range();
    acc.add(
        AssistId::refactor_rewrite("replace_qualified_name_with_use"),
        "Replace qualified path with use",
        target,
        |builder| {
            // Now that we've brought the name into scope, re-qualify all paths that could be
            // affected (that is, all paths inside the node we added the `use` to).
            let scope = builder.make_import_scope_mut(scope);
            shorten_paths(scope.as_syntax_node(), &original_path);
            let path = drop_generic_args(&original_path);
            let edition = ctx
                .sema
                .scope(original_path.syntax())
                .map(|semantics_scope| semantics_scope.krate().edition(ctx.db()))
                .unwrap_or(Edition::CURRENT);
            // stick the found import in front of the to be replaced path
            let path =
                match path_to_qualifier.and_then(|it| mod_path_to_ast(&it, edition).qualifier()) {
                    Some(qualifier) => make::path_concat(qualifier, path),
                    None => path,
                };
            insert_use(&scope, path, &ctx.config.insert_use);
        },
    )
}

fn drop_generic_args(path: &ast::Path) -> ast::Path {
    let path = path.clone_for_update();
    if let Some(segment) = path.segment() {
        if let Some(generic_args) = segment.generic_arg_list() {
            ted::remove(generic_args.syntax());
        }
    }
    path
}

/// Mutates `node` to shorten `path` in all descendants of `node`.
fn shorten_paths(node: &SyntaxNode, path: &ast::Path) {
    for child in node.children() {
        match_ast! {
            match child {
                // Don't modify `use` items, as this can break the `use` item when injecting a new
                // import into the use tree.
                ast::Use(_) => continue,
                // Don't descend into submodules, they don't have the same `use` items in scope.
                // FIXME: This isn't true due to `super::*` imports?
                ast::Module(_) => continue,
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
                        .is_some_and(|(lhs, rhs)| lhs.text() == rhs.text()) => {}
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
    fn assist_runs_on_first_segment() {
        check_assist(
            replace_qualified_name_with_use,
            r"
mod std { pub mod fmt { pub trait Debug {} } }
fn main() {
    $0std::fmt::Debug;
    let x: std::fmt::Debug = std::fmt::Debug;
}
    ",
            r"
use std::fmt;

mod std { pub mod fmt { pub trait Debug {} } }
fn main() {
    fmt::Debug;
    let x: fmt::Debug = fmt::Debug;
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

    #[test]
    fn replace_reuses_path_qualifier() {
        check_assist(
            replace_qualified_name_with_use,
            r"
pub mod foo {
    pub struct Foo;
}

mod bar {
    pub use super::foo::Foo as Bar;
}

fn main() {
    foo::Foo$0;
}
",
            r"
use foo::Foo;

pub mod foo {
    pub struct Foo;
}

mod bar {
    pub use super::foo::Foo as Bar;
}

fn main() {
    Foo;
}
",
        );
    }

    #[test]
    fn replace_does_not_always_try_to_replace_by_full_item_path() {
        check_assist(
            replace_qualified_name_with_use,
            r"
use std::mem;

mod std {
    pub mod mem {
        pub fn drop<T>(_: T) {}
    }
}

fn main() {
    mem::drop$0(0);
}
",
            r"
use std::mem::{self, drop};

mod std {
    pub mod mem {
        pub fn drop<T>(_: T) {}
    }
}

fn main() {
    drop(0);
}
",
        );
    }

    #[test]
    fn replace_should_drop_generic_args_in_use() {
        check_assist(
            replace_qualified_name_with_use,
            r"
mod std {
    pub mod mem {
        pub fn drop<T>(_: T) {}
    }
}

fn main() {
    std::mem::drop::<usize>$0(0);
}
",
            r"
use std::mem::drop;

mod std {
    pub mod mem {
        pub fn drop<T>(_: T) {}
    }
}

fn main() {
    drop::<usize>(0);
}
",
        );
    }
}
